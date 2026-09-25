# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import re
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class GQAIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        vit_transform,
        data_dir_list,
        num_used_data,
        parquet_info=None,
        image_dir_list=None,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
    ):
        self.image_dir_list = image_dir_list or []
        self._image_parquet_index = None
        super().__init__(
            dataset_name=dataset_name,
            transform=transform,
            tokenizer=tokenizer,
            vit_transform=vit_transform,
            data_dir_list=data_dir_list,
            num_used_data=num_used_data,
            parquet_info=parquet_info,
            image_dir_list=image_dir_list,
            local_rank=local_rank,
            world_size=world_size,
            num_workers=num_workers,
            data_status=data_status,
        )

    def _is_bytes_like(self, value):
        return isinstance(value, (bytes, bytearray, memoryview))

    def _get_row_value(self, row, key, default=None):
        try:
            return row[key]
        except Exception:
            return default

    def _load_embedded_image(self, row):
        # Prefer embedded parquet image payloads when available.
        for key in ("image", "image_bytes", "image_byte", "img", "image_data"):
            image_bytes = self._extract_image_bytes(self._get_row_value(row, key))
            if image_bytes is not None:
                return pil_img2rgb(Image.open(io.BytesIO(image_bytes)))
        return None

    def _normalize_image_id(self, image_id):
        image_id = str(image_id).strip()
        if image_id.isdigit():
            image_id = str(int(image_id))
        return image_id

    def _extract_image_bytes(self, value):
        if value is None:
            return None
        if isinstance(value, dict) and self._is_bytes_like(value.get("bytes")):
            return value["bytes"]
        if self._is_bytes_like(value):
            return value
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            if isinstance(first, dict) and self._is_bytes_like(first.get("bytes")):
                return first["bytes"]
            if self._is_bytes_like(first):
                return first
        return None

    def _resolve_image_path(self, image_id):
        image_id = str(image_id).strip()
        if image_id.isdigit():
            image_id = str(int(image_id))

        search_roots = [Path(p) for p in self.image_dir_list if p]
        if not search_roots:
            raise ValueError(
                "GQAIterableDataset could not find embedded image bytes in the parquet row, "
                "and no image_dir was provided as a fallback."
            )

        valid_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}
        candidates = []

        for root in search_roots:
            candidates.append(root / image_id)
            for suffix in valid_suffixes:
                candidates.append(root / f"{image_id}{suffix}")

            if image_id.isdigit():
                n = int(image_id)
                candidates.extend([
                    root / f"{n:012d}.jpg",
                    root / f"{n:012d}.jpeg",
                    root / f"{n:012d}.png",
                    root / f"000000{n}.jpg",
                    root / f"000000{n}.jpeg",
                    root / f"000000{n}.png",
                ])

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        for root in search_roots:
            for candidate in root.rglob("*"):
                if not candidate.is_file():
                    continue
                if candidate.suffix not in valid_suffixes:
                    continue

                name = candidate.name
                stem_digits = re.sub(r"\D", "", candidate.stem)

                if (
                    image_id in name
                    or stem_digits == image_id
                    or stem_digits.endswith(image_id)
                ):
                    return str(candidate)

        raise FileNotFoundError(
            f"Could not locate an image for GQA imageId={image_id} under {search_roots}"
        )

    def _iter_image_parquet_files(self):
        for root in self.image_dir_list:
            if not root:
                continue
            root = Path(root)
            if root.is_file() and root.suffix == ".parquet":
                yield root
            elif root.is_dir():
                yield from root.rglob("*.parquet")

    def _build_image_parquet_index(self):
        if self._image_parquet_index is not None:
            return self._image_parquet_index

        image_parquet_index = {}
        for parquet_path in self._iter_image_parquet_files():
            try:
                pf = pq.ParquetFile(parquet_path)
                schema_names = list(pf.schema.names)
                lower_to_name = {name.lower(): name for name in schema_names}
                image_id_col = (
                    lower_to_name.get("imageid")
                    or lower_to_name.get("image_id")
                    or lower_to_name.get("id")
                )
                image_col = (
                    lower_to_name.get("image")
                    or lower_to_name.get("image_bytes")
                    or lower_to_name.get("image_byte")
                    or lower_to_name.get("img")
                    or lower_to_name.get("image_data")
                )
                if image_id_col is None:
                    continue
                if image_col is None and "bytes" in lower_to_name and "path" in lower_to_name:
                    image_col = "image"
                if image_col is None:
                    non_id_cols = [
                        name for name in schema_names
                        if name != image_id_col and name.lower() != "path"
                    ]
                    if len(non_id_cols) == 1:
                        image_col = non_id_cols[0]
                    else:
                        continue

                for row_group_id in range(pf.num_row_groups):
                    table = pf.read_row_group(row_group_id, columns=[image_id_col, image_col])
                    rows = table.to_pydict()
                    image_ids = rows.get(image_id_col)
                    if image_ids is None:
                        continue

                    for row_idx, candidate_id in enumerate(image_ids):
                        normalized_id = self._normalize_image_id(candidate_id)
                        if normalized_id not in image_parquet_index:
                            image_parquet_index[normalized_id] = (
                                str(parquet_path),
                                row_group_id,
                                row_idx,
                                image_id_col,
                                image_col,
                            )
            except Exception:
                continue

        self._image_parquet_index = image_parquet_index
        return image_parquet_index

    def _load_image_from_parquet_sources(self, image_id):
        normalized_image_id = self._normalize_image_id(image_id)
        image_parquet_index = self._build_image_parquet_index()
        entry = image_parquet_index.get(normalized_image_id)
        if entry is None:
            raise FileNotFoundError(
                f"Could not locate an image for GQA imageId={normalized_image_id} in parquet image sources "
                f"{list(self._iter_image_parquet_files())}"
            )

        parquet_path, row_group_id, row_idx, image_id_col, image_col = entry
        pf = pq.ParquetFile(parquet_path)
        table = pf.read_row_group(row_group_id, columns=[image_id_col, image_col])
        rows = table.to_pydict()
        image_values = rows.get(image_col)
        if image_values is None or row_idx >= len(image_values):
            raise FileNotFoundError(
                f"Could not load image bytes for GQA imageId={normalized_image_id} from {parquet_path}"
            )

        image_bytes = self._extract_image_bytes(image_values[row_idx])
        if image_bytes is None:
            raise FileNotFoundError(
                f"Could not load image bytes for GQA imageId={normalized_image_id} from {parquet_path}"
            )

        return pil_img2rgb(Image.open(io.BytesIO(image_bytes)))

    def _load_image(self, image_id):
        try:
            image_path = self._resolve_image_path(image_id)
            return pil_img2rgb(Image.open(image_path))
        except (FileNotFoundError, ValueError):
            return self._load_image_from_parquet_sources(image_id)

    def parse_row(self, row):
        image = self._load_embedded_image(row)
        if image is None:
            image_id = self._get_row_value(row, "imageId")
            image = self._load_image(image_id)

        question = str(self._get_row_value(row, "question", "")).strip()
        answer = str(self._get_row_value(row, "fullAnswer", "")).strip()
        if not answer:
            answer = str(self._get_row_value(row, "answer", "")).strip()

        data = self._init_data()
        data = self._add_image(
            data,
            image,
            need_loss=False,
            need_vae=False,
            need_vit=True,
        )
        data = self._add_text(data, f"Question: {question}\nAnswer:", need_loss=False)
        data = self._add_text(data, answer, need_loss=True)
        return data
