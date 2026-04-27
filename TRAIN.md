# Training Guide

## 1. Data Preparation

### 1.1 Download Training Data

Training data is available on HuggingFace:

- **Link**: [https://huggingface.co/datasets/Co2y/UP2You_Dataset](https://huggingface.co/datasets/Co2y/UP2You_Dataset)

Download and extract all sub-directories from the dataset `UP2You_Dataset`.

### 1.2 Generate Data JSON Files

After downloading and extracting the data, you need to generate the JSON index files for training.

1. Open `get_data_json.py` and set `DATA_ROOT` to the path of your downloaded data folder:

```python
DATA_ROOT = "/path/to/your/downloaded/data"
```

2. Run the script to generate JSON files:

```bash
python get_data_json.py
```

This will create a `data/` directory containing the following JSON files:
- `2K2K.json`
- `4D-DRESS.json`
- `CustomHumans.json`
- `Human4DiT.json`
- `THuman2.1.json`
- `X_Humans.json`

## 2. Training

UP2You has three training stages, each using a different config file. Training is launched via `launch.py`.

### 2.1 Stage 1: Multi-view RGB Generation (i2mv)

Train the image-to-multi-view RGB generation model:

```bash
python launch.py --config configs/i2mv_sd21.yaml --train --gpu 0,1,2,3
```

### 2.2 Stage 2: Multi-view Normal Prediction (mv2normal)

Train the multi-view RGB to normal map prediction model:

```bash
python launch.py --config configs/mv2normal_sd21.yaml --train --gpu 0,1,2,3
```

### 2.3 Stage 3: Shape Prediction (mv2shape)

Train the shape (SMPL-X beta parameters) prediction model:

```bash
python launch.py --config configs/mv2shape.yaml --train --gpu 0,1,2,3
```

Alternatively, you can run all three stages sequentially using:

```bash
bash train.sh
```

### Training Options

| Option | Description |
|---|---|
| `--config` | Path to the YAML config file |
| `--train` | Run training |
| `--validate` | Run validation only |
| `--test` | Run testing only |
| `--gpu` | GPU IDs to use (e.g., `0,1,2,3`) |
| `--wandb` | Enable Weights & Biases logging |

### Config Customization

You can customize training hyperparameters by editing the config YAML files in `configs/`:

- **`i2mv_sd21.yaml`**: Multi-view RGB generation config (learning rate, batch size, number of views, etc.)
- **`mv2normal_sd21.yaml`**: Normal prediction config
- **`mv2shape.yaml`**: Shape prediction config

Key parameters:
- `data.batch_size`: Training batch size per GPU
- `data.num_workers`: Number of data loading workers
- `system.optimizer.args.lr`: Learning rate
- `trainer.max_epochs`: Maximum training epochs
- `trainer.strategy`: Training strategy (`ddp` for multi-GPU)
- `trainer.precision`: Training precision (`bf16-mixed` recommended)

## 3. Model Outputs

Training outputs (checkpoints, logs, visualizations) are saved to the `outputs/` directory by default, organized as:

```
outputs/
├── i2mv/
│   └── <tag>/
│       ├── ckpts/          # Model checkpoints
│       ├── tb_logs/        # TensorBoard logs
│       ├── save/           # Validation/test visualizations
│       └── weights/        # Exported model weights
├── mv2normal/
│   └── <tag>/
│       └── ...
└── mv2shape/
    └── <tag>/
        └── ...
```
