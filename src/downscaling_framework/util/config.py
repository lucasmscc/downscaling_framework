from dataclasses import dataclass, field

@dataclass
class GeneralConfig:
    random_state: int = 42
    device: str = "cuda"

@dataclass
class DataConfig:
    n_points: int = 64
    test_size: float = 0.2
    batch_size: int = 128

@dataclass
class TrainingConfig:
    epochs: int = 50
    n_trials: int = 20
    patience: int = 5
    min_delta: float = 1e-5

@dataclass
class ModelConfig:
    min_seq_len: int = 12
    max_seq_len: int = 48
    step: int = 12
    lr_range: list[float] = field(default_factory=lambda: [1e-4, 1e-2])

@dataclass
class AppConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models: dict[str, ModelConfig] = field(default_factory=lambda: {
        "cnn": ModelConfig(),
        "lstm": ModelConfig(),
        "elasticnet": ModelConfig(min_seq_len=0, max_seq_len=0, step=0),
        "mlp": ModelConfig(min_seq_len=0, max_seq_len=0, step=0)
    })
CONFIG = AppConfig()