# Sensor Context Encoder Challenge

This repository contains a compact PyTorch implementation for the UCI HAR smartphone activity-recognition challenge, with:

- a direct sensor-classification baseline
- a context-embedding model that projects sensor windows into the embedding space of a frozen `HuggingFaceTB/SmolLM2-360M-Instruct`
- a shuffled-embedding evaluation for the sensor-dependence check

The task uses the 9 raw inertial-signal channels from the UCI HAR dataset to predict 6 activity classes.

## Project Structure

```text
.
├── har_classifier/
│   ├── config.py
│   ├── context_eval_runner.py
│   ├── context_model.py
│   ├── context_runner.py
│   ├── data.py
│   ├── engine.py
│   ├── llm_utils.py
│   ├── models.py
│   ├── runner.py
│   ├── shuffle_eval.py
│   └── utils.py
├── train.py
├── train_context.py
├── evaluate_context_shuffle.py
├── train/
├── test/
├── activity_labels.txt
├── features.txt
├── features_info.txt
└── README.txt
```

## What Each Entry Point Does

- `train.py`
  Trains the direct baseline:
  `sensor window -> sensor encoder -> linear classification head`

- `train_context.py`
  Trains the context model:
  `sensor window -> sensor encoder -> projector -> frozen SmolLM2 -> linear head`

- `evaluate_context_shuffle.py`
  Re-runs the trained context model after shuffling complete projected sensor embeddings across test examples.

## Environment

This project was developed in a Conda environment named `py12`.

Recommended package versions:

- `python 3.12`
- `torch 2.2.2`
- `transformers 4.49.0`

`transformers 5.x` is not recommended here because it caused backend compatibility issues with `torch 2.2.2` in the local environment on August 24, 2026.

## Setup

Create or activate your environment:

```bash
conda activate py12
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

This repository is currently organized directly inside the UCI HAR dataset root. The code expects this layout:

```text
UCI HAR Dataset/
├── train/
│   ├── Inertial Signals/
│   ├── X_train.txt
│   ├── y_train.txt
│   └── subject_train.txt
├── test/
│   ├── Inertial Signals/
│   ├── X_test.txt
│   ├── y_test.txt
│   └── subject_test.txt
├── activity_labels.txt
├── features.txt
├── features_info.txt
├── train.py
└── har_classifier/
```

The implementation uses only the 9 raw inertial-signal files from `train/Inertial Signals/` and `test/Inertial Signals/`.

## Run Commands

### 1. Direct Baseline

```bash
python train.py
```

Optional example:

```bash
python train.py --epochs 10 --batch-size 64
```

### 2. Context-Embedding Model

```bash
python train_context.py
```

Recommended on Apple Silicon / MPS:

```bash
python train_context.py --llm-dtype float32
```

Optional example:

```bash
python train_context.py --epochs 5 --batch-size 16 --llm-dtype float32
```

### 3. Shuffled-Embedding Check

After training the context model, run:

```bash
python evaluate_context_shuffle.py --checkpoint-path context_embedding_classifier.pt
```

## Important Implementation Notes

- The direct baseline and context model use the same subject-wise train/validation/test split logic.
- The language model backbone is frozen, including the token-embedding table.
- The sensor window is inserted as a continuous embedding through `inputs_embeds`, not converted into text.
- The context-model code includes a trainable-parameter budget check for the sensor encoder, projector, and classification head.

## Notes

- `README.txt` is the original dataset documentation from UCI HAR.
- The project currently lives inside the dataset directory for convenience. If you later want a cleaner public repo, we can move the code into a separate project root and keep the dataset as an external download step.
