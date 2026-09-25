# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import traceback
from pathlib import Path
from PIL import Image, ImageFile, PngImagePlugin

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


_INDEX_SCALAR_TYPES = (str, int, float, bool, type(None))

def _resolve_media_path(image_dir, media_name):
    """Resolve a relative media name against the configured image directory."""
    path = Path(str(media_name))
    if not path.is_absolute():
        return str(Path(image_dir) / path)
    if path.exists():
        return str(path)
    return str(path)


def _get_nested_value(data_item, field):
    if field == "gpt_answer":
        for conversation in data_item.get("conversations", []):
            if isinstance(conversation, dict) and conversation.get("from") == "gpt":
                return conversation.get("value")
        return None
    if field in data_item:
        return data_item[field]
    current = data_item
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _json_safe_index_value(value):
    if isinstance(value, _INDEX_SCALAR_TYPES):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, _INDEX_SCALAR_TYPES) for item in value
    ):
        return list(value)
    return None


def _copy_data_index_fields(data_item, fields):
    copied = {}
    for field in fields:
        value = _json_safe_index_value(_get_nested_value(data_item, field))
        if value is not None:
            copied[field] = value
            normalized = field.replace(".", "_")
            if normalized != field:
                copied.setdefault(normalized, value)
    return copied


class SftJSONLIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, frame_sampler, 
        jsonl_path_list, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=False, shuffle_seed=0, vit_transform=None,
        source_dataset_names=None, data_index_fields=None,
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform or transform
        self.tokenizer = tokenizer
        self.frame_sampler = frame_sampler
        self.data_status = data_status
        self.data_index_fields = tuple(data_index_fields or ())
        self.data_paths = self.get_data_paths(
            jsonl_path_list, 
            data_dir_list, 
            num_used_data, 
            shuffle_lines, 
            shuffle_seed,
            source_dataset_names,
        )
        self.set_epoch()

    def get_data_paths(
        self, 
        jsonl_path_list, 
        data_dir_list, 
        num_used_data, 
        shuffle_lines, 
        shuffle_seed,
        source_dataset_names,
    ):
        data_paths = []
        if source_dataset_names is None:
            source_dataset_names = [None] * len(jsonl_path_list)
        if len(source_dataset_names) != len(jsonl_path_list):
            raise ValueError(
                "source_dataset_names must align with jsonl_path_list: "
                f"{len(source_dataset_names)} != {len(jsonl_path_list)}"
            )
        for jsonl_path, image_dir, num_data_point, source_dataset_name in zip(
            jsonl_path_list, data_dir_list, num_used_data, source_dataset_names
        ):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            raw_data = raw_data[:num_data_point]
            data_paths.extend(
                [(json_data, image_dir, source_dataset_name) for json_data in raw_data]
            )
        return data_paths

    def change_format(self, data, num_images):
        elements = []
        for conversation in data['conversations']:
            if conversation['from'] == 'human':
                if '<image>' not in conversation['value']:
                    elements.append({
                        'type': 'text',
                        'has_loss': 0,
                        'text': conversation['value'],
                    })
                else:
                    text_list = conversation['value'].split('<image>')
                    for idx, text in enumerate(text_list):
                        if text.strip() != '':
                            elements.append({
                                'type': 'text',
                                'has_loss': 0,
                                'text': text.strip(),
                            })
                        if (idx != len(text_list) - 1) and (idx < num_images):
                            elements.append({'type': 'image',})
            elif conversation['from'] == 'gpt':
                elements.append({
                    'type': 'text',
                    'has_loss': 1,
                    'text': conversation['value'],
                })
        return elements

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0
        transform_stride = self.vit_transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, image_dir, source_dataset_name) in enumerate(data_paths_per_worker_, start=row_start_id):
                num_tokens = 0
                image_tensor_list = []
                text_ids_list = []
                sequence_plan = []

                try:
                    data_item = json.loads(data)
                    raw_images = None
                    if data_item.get('image') is not None:
                        if type(data_item['image']) == list:
                            raw_images = [
                                pil_img2rgb(
                                    Image.open(
                                        _resolve_media_path(image_dir, image)
                                    )
                                )
                                for image in data_item['image']
                            ]
                        else:
                            raw_images = [
                                pil_img2rgb(
                                    Image.open(
                                        _resolve_media_path(
                                            image_dir, data_item['image']
                                        )
                                    )
                                )
                            ]
                    elif 'video' in data_item:
                        raw_images = self.frame_sampler(
                            _resolve_media_path(image_dir, data_item['video'])
                        )
                        special_tokens = '<image>' * len(raw_images)
                        for item in data_item['conversations']:
                            if '<video>' in item['value']:
                                item['value'] = item['value'].replace('<video>', special_tokens)
                                break
                            else:
                                raise ValueError("Cannot find <video> in the conversation!")
                except:
                    traceback.print_exc()
                    continue

                if raw_images:
                    for raw_image in raw_images:
                        image_tensor = self.vit_transform(raw_image, img_num=len(raw_images))
                        image_tensor_list.append(image_tensor)
                        height, width = image_tensor.shape[1:]
                        num_tokens += width * height // transform_stride ** 2

                elements = self.change_format(data_item, len(image_tensor_list))

                for item in elements:
                    if item['type'] == 'text':
                        text_data = item['text']
                        text_ids = self.tokenizer.encode(text_data)
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += len(text_ids)
                            current_plan = {
                                'type': 'text',
                                'enable_cfg': 0,
                                'loss': item['has_loss'],
                                'special_token_loss': 0,
                                'special_token_label': None,
                            }
                            sequence_plan.append(current_plan)
                    elif item['type'] == 'image':
                        current_plan = {
                            'type': 'vit_image',
                            'enable_cfg': 0,
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        }
                        sequence_plan.append(current_plan)

                has_loss = [item['loss'] for item in sequence_plan]
                if sum(has_loss) == 0:
                    print(f'No loss defined, skipped.')
                    continue

                data_indexes = {
                    "data_indexes": row_idx,
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                    "dataset_group_name": self.dataset_name,
                    "source_dataset_name": source_dataset_name or self.dataset_name,
                }
                if self.data_index_fields:
                    data_indexes.update(
                        _copy_data_index_fields(data_item, self.data_index_fields)
                    )

                yield dict(
                    image_tensor_list=image_tensor_list,
                    text_ids_list=text_ids_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    data_indexes=data_indexes,
                )

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
