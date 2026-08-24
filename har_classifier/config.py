from dataclasses import asdict, dataclass
from pathlib import Path

SIGNAL_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

CLASS_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

SEQUENCE_LENGTH = 128
NUM_CHANNELS = 9
ENCODER_OUTPUT_DIM = 256
DEFAULT_LLM_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
SENSOR_PLACEHOLDER = "<SENSOR>"
DEFAULT_CONTEXT_PROMPT = (
    "Classify the activity as walking, walking upstairs, walking downstairs, "
    "sitting, standing, or laying.\n\n"
    f"Sensor context: {SENSOR_PLACEHOLDER}\n\n"
    "Activity:"
)


@dataclass(slots=True)
class TrainingConfig:
    base_dir: Path
    seed: int = 42
    batch_size: int = 64
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_subject_fraction: float = 0.2
    num_workers: int = 0
    num_classes: int = 6
    checkpoint_path: str = "direct_sensor_classifier.pt"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["base_dir"] = str(self.base_dir)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        config_data = dict(data)
        config_data["base_dir"] = Path(config_data["base_dir"])
        return cls(**config_data)


@dataclass(slots=True)
class ContextTrainingConfig:
    base_dir: Path
    seed: int = 42
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    val_subject_fraction: float = 0.2
    num_workers: int = 0
    num_classes: int = 6
    checkpoint_path: str = "context_embedding_classifier.pt"
    llm_model_name: str = DEFAULT_LLM_MODEL_NAME
    prompt_template: str = DEFAULT_CONTEXT_PROMPT
    sensor_placeholder: str = SENSOR_PLACEHOLDER
    projector_hidden_dim: int = 512
    llm_dtype: str = "auto"
    max_trainable_params: int = 10_000_000
    run_shuffle_check: bool = True
    shuffle_seed: int = 43

    def to_dict(self) -> dict:
        data = asdict(self)
        data["base_dir"] = str(self.base_dir)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ContextTrainingConfig":
        config_data = dict(data)
        config_data["base_dir"] = Path(config_data["base_dir"])
        return cls(**config_data)
