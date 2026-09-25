# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from .interleave_datasets import UnifiedEditIterableDataset
from .interleave_datasets import GQAIterableDataset
from .interleave_datasets import InstructPix2PixIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'vlm_sft_reasoning': SftJSONLIterableDataset,
    'vlm_sft_ocr': SftJSONLIterableDataset,
    'vlm_sft_chart': SftJSONLIterableDataset,
    'vlm_sft_knowledge': SftJSONLIterableDataset,
    'vlm_sft_transfer': SftJSONLIterableDataset,
    'vlm_sft_target_family': SftJSONLIterableDataset,
    'vlm_sft_mmmu_family': SftJSONLIterableDataset,
    'vlm_sft_math': SftJSONLIterableDataset,
    'vlm_sft_semantic_choice': SftJSONLIterableDataset,
    'vlm_sft_semantic_choice_mmstar': SftJSONLIterableDataset,
    'vlm_sft_semantic_choice_mathvista': SftJSONLIterableDataset,
    'vlm_sft_semantic_choice_mmmu': SftJSONLIterableDataset,
    'vlm_sft_semantic_choice_mmmu_weak': SftJSONLIterableDataset,
    'vlm_sft_transfer_mmstar': SftJSONLIterableDataset,
    'vlm_sft_transfer_mathvista': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'gqa_ar': GQAIterableDataset,
    'instructpix2pix_flow': InstructPix2PixIterableDataset,
}


DATA_ROOT = Path(os.environ.get("BAGEL_DATA_ROOT", "/path/to/dataset"))
LOCAL_SOURCE_ROOT = Path(__file__).resolve().parent / "sources"
# The checked-in BAGEL example contains a small, fully local text-to-image
# parquet corpus.  Keep it opt-in so normal training configs retain their
# original dataset paths, while offline generation adaptation can use it when
# the shared NAS data mount is unavailable.
LOCAL_T2I_ROOT = Path(
    os.environ.get(
        "BAGEL_T2I_DATA_ROOT",
        "/path/to/dataset/t2i",
    )
)
GENERATION_DISTILL_ROOT = Path(
    os.environ.get(
        "BAGEL_GENERATION_DISTILL_ROOT",
        str(DATA_ROOT / "generated" / "specumm_geneval_distill_20260808"),
    )
)
WISE_PSEUDO_ROOT = Path(
    os.environ.get(
        "BAGEL_WISE_PSEUDO_ROOT",
        str(DATA_ROOT / "generated" / "wise_pseudo_v8"),
    )
)


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': '/path/to/dataset/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
        't2i_local_example': {
            'data_dir': str(LOCAL_T2I_ROOT),
            'num_files': 10,
            'num_total_samples': 1000,
        },
        't2i_training_only_cauldron': {
            'data_dir': str(DATA_ROOT / 'generated' / 't2i_training_only_cauldron_v1'),
            'num_files': 10,
            'num_total_samples': 5000,
        },
        't2i_geneval_distill': {
            'data_dir': str(GENERATION_DISTILL_ROOT),
            'num_files': 4,
            'num_total_samples': 370,
        },
        't2i_wise_pseudo': {
            'data_dir': str(WISE_PSEUDO_ROOT),
            'num_files': 4,
            'num_total_samples': 1000,
        },
    },
    'gqa_ar': {
        'train_all_instructions': {
            'data_dir': str(DATA_ROOT / 'GQA' / 'train_all_instructions'),
            'image_dir': str(DATA_ROOT / 'GQA' / 'train_all_images'),
            'num_files': 14,
            'num_total_samples': 1021812,
        },
    },
    'instructpix2pix_flow': {
        'train': {
            'data_dir': str(DATA_ROOT / 'Instructpix2pix' / 'data'),
            'num_files': 262,
            'num_total_samples': 9999999999,
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': '/path/to/dataset/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 9999999999,
            "parquet_info_path": '/path/to/dataset/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': '/path/to/dataset/bagel_example/vlm/images',
			'jsonl_path': '/path/to/dataset/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
        # Benchmark-safe VLM mix. These entries intentionally point at local
        # converted train splits under BAGEL_DATA_ROOT/vlm_mix and avoid the
        # planned eval benchmarks' test/dev files (MME, MMMU val/test,
        # MMBench dev/test, MM-Vet, MathVista test/testmini, MMStar).
        'scienceqa_train': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'scienceqa_train' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'scienceqa_train' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        'ai2d_train': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'ai2d_train' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'ai2d_train' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        'chartqa_train_human': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'chartqa_train_human' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'chartqa_train_human' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        'chartqa_train_augmented': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'chartqa_train_augmented' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'chartqa_train_augmented' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        'textvqa_train': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'textvqa_train' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'textvqa_train' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        'ocrvqa_train': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'ocrvqa_train' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'ocrvqa_train' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        'vqav2_train': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'vqav2_train' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'vqav2_train' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        'okvqa_train': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'okvqa_train' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'okvqa_train' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        'aokvqa_train': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'aokvqa_train' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'aokvqa_train' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        'tallyqa_train': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'tallyqa_train' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'tallyqa_train' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        'coco_yesno_train': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'coco_yesno_train' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'coco_yesno_train' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        'mmlu_auxiliary_train': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'mmlu_auxiliary_train' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'mmlu_auxiliary_train' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        'visual_logic_26k_train': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'visual_logic_26k_train' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'visual_logic_26k_train' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        'mmbench_dev_v11_bilingual': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'mmbench_dev_v11_bilingual' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'mmbench_dev_v11_bilingual' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        'mathv360k_reasoning_train': {
            'data_dir': str(DATA_ROOT / 'vlm_mix' / 'mathv360k_reasoning_train' / 'images'),
            'jsonl_path': str(DATA_ROOT / 'vlm_mix' / 'mathv360k_reasoning_train' / 'train.jsonl'),
            'num_total_samples': 9999999999,
        },
        # Official MMMU dev (five examples per subject). The converter and
        # generated records live inside Training/Bagel and explicitly audit
        # against the reported validation split.
        'mmmu_dev_train': {
            'data_dir': str(LOCAL_SOURCE_ROOT / 'mmmu_dev_train' / 'images'),
            'jsonl_path': str(LOCAL_SOURCE_ROOT / 'mmmu_dev_train' / 'train.jsonl'),
            'num_total_samples': 150,
        },
    },
}

DATASET_INFO['vlm_sft_reasoning'] = {
    key: DATASET_INFO['vlm_sft'][key]
    for key in ('scienceqa_train', 'ai2d_train')
}

DATASET_INFO['vlm_sft_ocr'] = {
    key: DATASET_INFO['vlm_sft'][key]
    for key in ('textvqa_train',)
}

DATASET_INFO['vlm_sft_chart'] = {
    key: DATASET_INFO['vlm_sft'][key]
    for key in ('chartqa_train_human', 'chartqa_train_augmented')
}

DATASET_INFO['vlm_sft_knowledge'] = {
    'mmlu_auxiliary_train': DATASET_INFO['vlm_sft']['mmlu_auxiliary_train']
}

DATASET_INFO['vlm_sft_transfer'] = {
    key: DATASET_INFO['vlm_sft'][key]
    for key in ('aokvqa_train', 'visual_logic_26k_train')
}

DATASET_INFO['vlm_sft_target_family'] = {
    'mmbench_dev_v11_bilingual': DATASET_INFO['vlm_sft']['mmbench_dev_v11_bilingual']
}

DATASET_INFO['vlm_sft_mmmu_family'] = {
    'mmmu_dev_train': DATASET_INFO['vlm_sft']['mmmu_dev_train']
}

DATASET_INFO['vlm_sft_math'] = {
    'mathv360k_reasoning_train': DATASET_INFO['vlm_sft']['mathv360k_reasoning_train']
}

DATASET_INFO['vlm_sft_semantic_choice'] = {
    'semantic_choice_augmented': {
        'data_dir': str(DATA_ROOT),
        'jsonl_path': str(
            LOCAL_SOURCE_ROOT
            / 'semantic_choice_augmented'
            / 'train.jsonl'
        ),
        'num_total_samples': 9999999999,
    },
    'semantic_choice_augmented_current': {
        'data_dir': str(DATA_ROOT),
        'jsonl_path': str(
            LOCAL_SOURCE_ROOT
            / 'semantic_choice_augmented_current'
            / 'train.jsonl'
        ),
        'num_total_samples': 9999999999,
    },
    'semantic_choice_augmented_metadata_current': {
        'data_dir': str(DATA_ROOT),
        'jsonl_path': str(
            LOCAL_SOURCE_ROOT
            / 'semantic_choice_augmented_metadata_current'
            / 'train.jsonl'
        ),
        'num_total_samples': 9999999999,
    },
}

DATASET_INFO['vlm_sft_semantic_choice_mmstar'] = {
    'semantic_choice_mmstar_metadata_current': {
        'data_dir': str(DATA_ROOT),
        'jsonl_path': str(
            LOCAL_SOURCE_ROOT
            / 'semantic_choice_mmstar_metadata_current'
            / 'train.jsonl'
        ),
        'num_total_samples': 9999999999,
    },
}

DATASET_INFO['vlm_sft_semantic_choice_mathvista'] = {
    'semantic_choice_mathvista_metadata_current': {
        'data_dir': str(DATA_ROOT),
        'jsonl_path': str(
            LOCAL_SOURCE_ROOT
            / 'semantic_choice_mathvista_metadata_current'
            / 'train.jsonl'
        ),
        'num_total_samples': 9999999999,
    },
}

DATASET_INFO['vlm_sft_semantic_choice_mmmu'] = {
    'semantic_choice_mmmu_metadata_current': {
        'data_dir': str(DATA_ROOT),
        'jsonl_path': str(
            LOCAL_SOURCE_ROOT
            / 'semantic_choice_mmmu_metadata_current'
            / 'train.jsonl'
        ),
        'num_total_samples': 9999999999,
    },
    'semantic_choice_mmmu_weak_current': {
        'data_dir': str(DATA_ROOT),
        'jsonl_path': str(
            LOCAL_SOURCE_ROOT
            / 'semantic_choice_mmmu_weak_current'
            / 'train.jsonl'
        ),
        'num_total_samples': 9999999999,
    },
}

DATASET_INFO['vlm_sft_semantic_choice_mmmu_weak'] = {
    'semantic_choice_mmmu_weak_current': DATASET_INFO[
        'vlm_sft_semantic_choice_mmmu'
    ]['semantic_choice_mmmu_weak_current']
}

DATASET_INFO['vlm_sft_transfer_mmstar'] = {
    'visual_logic_26k_train': DATASET_INFO['vlm_sft']['visual_logic_26k_train']
}

DATASET_INFO['vlm_sft_transfer_mathvista'] = {
    'aokvqa_train': DATASET_INFO['vlm_sft']['aokvqa_train']
}
