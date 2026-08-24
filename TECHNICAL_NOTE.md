# Technical Note: Sensor Context Encoder Challenge

## Summary

This project tests whether a continuous sensor embedding can act as a useful interface to a frozen language model for human activity recognition on the UCI HAR dataset. We compare three conditions: a direct sensor classifier, a frozen-LLM context-embedding model, and the same context model after shuffling complete projected embeddings across test examples.

## Data

- Dataset: UCI Human Activity Recognition Using Smartphones.
- Input: only the 9 inertial-signal channels from `Inertial Signals`.
- Input shape: `128 x 9` per example.
- Labels: walking, walking upstairs, walking downstairs, sitting, standing, laying.
- Split: official subject-wise train/test split; validation subjects are sampled only from the training split.
- Excluded: the 561-dimensional engineered feature vectors.

## Models

### Direct baseline

The baseline uses the same sensor encoder as the context model, followed by a linear classifier. The encoder is a 3-layer 1D CNN over the 9 input channels:

`[9 -> 64 -> 128 -> 256]`

Layer details:

- `Conv1d(9, 64, kernel_size=5, stride=1, padding=2)` + `BatchNorm1d(64)` + `ReLU`
- `Conv1d(64, 128, kernel_size=5, stride=1, padding=2)` + `BatchNorm1d(128)` + `ReLU`
- `Conv1d(128, 256, kernel_size=3, stride=1, padding=1)` + `BatchNorm1d(256)` + `ReLU`
- `AdaptiveAvgPool1d(1)` to produce one 256-dim feature vector per window
- `Linear(256, 6)` classification head

### Context-embedding model

Pipeline:

`sensor window -> sensor encoder -> projector -> frozen SmolLM2 -> linear head`

Implementation details:

- Language model: `HuggingFaceTB/SmolLM2-360M-Instruct`.
- All LM parameters are frozen, including the token-embedding table.
- The projected sensor vector is inserted directly with `inputs_embeds`, not converted to text.
- The prompt is:

```text
Classify the activity as walking, walking upstairs, walking downstairs, sitting, standing, or laying.

Sensor context: <SENSOR>

Activity:
```

- The classifier reads the final hidden state after `Activity:` and predicts the 6 classes with a trainable linear head.

### Sensor-dependence check

The shuffled-embedding evaluation reuses the trained context model but randomly permutes complete projected embeddings across test examples. This preserves realistic embeddings while breaking the sensor-to-label alignment. The model is not retrained.

## Training setup

- Optimizer: AdamW.
- Metric: macro-F1.
- Model selection: best validation macro-F1 checkpoint.
- Seeded training and evaluation.
- Validation split comes only from training subjects.
- Default seeds and hyperparameters in the code:
  - Seed: `42`
  - Direct baseline batch size: `64`
  - Context batch size: `16`
  - Direct baseline epochs: `30`
  - Context epochs: `10`
  - Validation fraction: `0.2`
  - Learning rate:
    - Direct baseline: `1e-3`
    - Context model: `5e-4`

## Trainable parameters

Only these components are trainable in the context model:

- Sensor encoder
- Projector
- Classification head

The code enforces a default upper bound of `10,000,000` trainable parameters for those parts.

## Limitations

- The context model is much heavier than the direct baseline because it includes a frozen pretrained LLM.
- Results can vary with the subject split and seed.
- The shuffle check is a single reproducible diagnostic, not a full causal proof.
- If the direct baseline already performs well, the context path may not justify the extra complexity.

## Recommendation

The context-embedding approach is a useful proof of concept for continuous sensor-to-language-model interfaces, but it should only be developed further if it shows a clear performance gain or at least a meaningful dependence on matched sensor embeddings.

## Results Table

| Condition | Macro-F1 |
| --- | --- |
| Direct sensor classifier |  |
| Context-embedding model |  |
| Context model with shuffled embeddings |  |
