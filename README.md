# Sensor Context Encoder Challenge

This repository contains an implementation of the UCI HAR sensor-context challenge with three required conditions:

1. a direct sensor-classification baseline
2. a context-embedding model using a frozen `HuggingFaceTB/SmolLM2-360M-Instruct`
3. a shuffled-embedding evaluation for the sensor-dependence check

The task uses the 9 raw inertial-signal channels from the UCI Human Activity Recognition Using Smartphones dataset to predict 6 activity classes.

## Repository Structure

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
├── TECHNICAL_NOTE.md
├── requirements.txt
├── train/
├── test/
├── activity_labels.txt
├── features.txt
├── features_info.txt
└── README.txt
```

## Environment

The project was developed in a Conda environment named `py12`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Layout

The code expects to run from the UCI HAR dataset root:

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

Only the 9 raw inertial-signal files in `train/Inertial Signals/` and `test/Inertial Signals/` are used. The 561 engineered features are not used.

## Reproducing the Reported Results

The results in `TECHNICAL_NOTE.md` were produced with the default code settings and `seed=42`.

### 1. Direct Baseline

```bash
python train.py --seed 42
```

### 2. Context-Embedding Model

On Apple Silicon / MPS, the stable setting used for the reported run was:

```bash
python train_context.py --seed 42 --llm-dtype float32
```

### 3. Shuffled-Embedding Check

```bash
python evaluate_context_shuffle.py \
  --checkpoint-path context_embedding_classifier.pt \
  --shuffle-seed 43
```

## Model Summary

### Direct baseline

```text
sensor window -> sensor encoder -> linear classification head
```

### Context model

```text
sensor window -> sensor encoder -> projector -> frozen SmolLM2 -> linear head
```

Implementation details:

- The language model backbone is frozen, including the token-embedding table.
- The sensor embedding is inserted through `inputs_embeds`, not converted into text.
- The context model uses the prompt:

```text
Classify the activity as walking, walking upstairs, walking downstairs, sitting, standing, or laying.

Sensor context: <SENSOR>

Activity:
```

## Notes

- `README.txt` is the original dataset documentation from UCI HAR.
- Additional implementation details, results, and conclusions are documented in `TECHNICAL_NOTE.md`.
