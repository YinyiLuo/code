# training

```bash
cd codebase
pip install -r requirements.txt
BAGEL_DATA_ROOT=/path/to/dataset MODEL_PATH=/path/to/BAGEL-7B-MoT NUM_GPUS=4 TOTAL_STEPS=1000
bash train.sh
```

The launcher initializes from pretrained BAGEL, trains the TAFE specialized
FFN banks, and trains understanding/generation early-exit routers with
multi-exit CE and hidden-state distillation. Checkpoints go to
`results/training/checkpoints/`.
