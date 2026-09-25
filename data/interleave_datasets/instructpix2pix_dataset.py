# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io

from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class InstructPix2PixIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    def __init__(self, *args, need_vit_condition=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.need_vit_condition = need_vit_condition

    def parse_row(self, row):
        original_image = pil_img2rgb(Image.open(io.BytesIO(row["original_image"]["bytes"])))
        edited_image = pil_img2rgb(Image.open(io.BytesIO(row["edited_image"]["bytes"])))
        edit_prompt = str(row["edit_prompt"]).strip()

        if not edit_prompt:
            return {}

        data = self._init_data()
        data = self._add_image(
            data,
            original_image,
            need_loss=False,
            need_vae=True,
            need_vit=self.need_vit_condition,
        )
        data = self._add_text(data, edit_prompt, need_loss=False)
        data = self._add_image(
            data,
            edited_image,
            need_loss=True,
            need_vae=False,
            need_vit=False,
        )
        return data
