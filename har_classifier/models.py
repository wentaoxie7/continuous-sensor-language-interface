import torch
import torch.nn as nn

from .config import ENCODER_OUTPUT_DIM, NUM_CHANNELS


class SensorEncoder(nn.Module):
    """Encode 9-channel inertial signals from [B, 128, 9] to [B, 256]."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(NUM_CHANNELS, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, ENCODER_OUTPUT_DIM, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(ENCODER_OUTPUT_DIM),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.network(x)
        return x.squeeze(-1)


class DirectSensorClassifier(nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.sensor_encoder = SensorEncoder()
        self.classifier = nn.Linear(ENCODER_OUTPUT_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.sensor_encoder(x)
        return self.classifier(features)
