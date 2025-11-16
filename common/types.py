from typing import Any
from enum import Enum
from dataclasses import dataclass

class Model(str, Enum):
    NANOGPT = 'nanogpt'

class Optimizer(str, Enum):
    ADAMW = 'adamw'

class Scheduler(str, Enum):
    COSINE_ANNEALING_LR = 'cosine_annealing_lr'

class Dataset(str, Enum):
    ENWIK8 = 'enwik8'

class TokenEncoding(str, Enum):
    GPT2 = 'gpt2'
    
@dataclass
class TrainArgs:
    model_name: Model
    dataset_name: Dataset
    batch_size: int
    gradient_accumulation_steps: int
    grad_clip: float
    vocab_size: int
    seq_len: int
    embed_dim: int
    hidden_dim: int
    num_transformer_blocks: int
    num_heads: int
    dropout_value: float
    use_bias: bool
    use_flash_attn: bool
    optimizer_name: Optimizer
    scheduler_name: Scheduler
    scheduler_max_steps: int
    lr: float
    lr_min: float
    weight_decay: float
    beta1: float
    beta2: float
    train_steps: int
    eval_steps: int
    eval_interval: int
    checkpoint_interval: int
    compile: bool
    use_cuda: bool

@dataclass
class SampleArgs:
    checkpoint_file_name: str
    prompt: str
    temperature: float
    top_k: int | None
    max_new_tokens: int
    compile: bool
    use_cuda: bool

@dataclass
class Checkpoint:
    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    scaler: dict[str, Any]
    args: TrainArgs
    step_number: int
    val_loss: float
