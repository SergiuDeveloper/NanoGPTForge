import os
import inspect
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from typing import Iterator, cast
from contextlib import nullcontext

from common.types import TrainArgs, SampleArgs, Checkpoint
from common.constants import OptimizerType, SchedulerType

def build_split_dataloader(
    file_path: str,
    batch_size: int,
    seq_len: int,
    shuffle: bool,
    pin_memory: bool,
    num_workers: int
) -> DataLoader[tuple[Tensor, Tensor]]:
    print(f'Loading dataset {file_path}')
    
    data = torch.from_numpy(np.load(file_path)).long()
    x = data[:-1].unfold(0, seq_len, 1)
    y = data[1:].unfold(0, seq_len, 1)

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
    print(f'Loaded dataset {file_path}')
    return cast(DataLoader[tuple[Tensor, Tensor]], dataloader)

def get_next_batch(
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    dataloader_iter: Iterator[tuple[Tensor, Tensor]],
    device: torch.device
) -> tuple[tuple[Tensor, Tensor], Iterator[tuple[Tensor, Tensor]]]:
    try:
        x, y = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        x, y = next(dataloader_iter)
    x, y = x.to(device), y.to(device)
    return (x, y), dataloader_iter

def build_optimizer(
    model: nn.Module,
    device: torch.device,
    optimizer_type: OptimizerType,
    scheduler_type: SchedulerType,
    lr: float,
    lr_min: float,
    betas: tuple[float, float],
    weight_decay: float,
    scheduler_max_steps: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer_groups = [
            { 'params': [param for param in params if param.dim() >= 2], 'weight_decay': weight_decay },
            { 'params': [param for param in params if param.dim() < 2], 'weight_decay': 0.0 }
        ]
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device.type == 'cuda'
    optimizer = optimizer_type(optimizer_groups, lr=lr, betas=betas, fused=use_fused)
    scheduler = scheduler_type(optimizer, T_max=scheduler_max_steps, eta_min=lr_min)
    return (optimizer, scheduler)

@torch.no_grad()
def compute_loss(
    model: nn.Module,
    dataloaders: list[DataLoader],
    steps: int,
    context: torch.autocast | nullcontext[None],
    device: torch.device
) -> list[float]:
    all_losses: list[float] = []
    model.eval()
    for dataloader in dataloaders:
        losses = torch.zeros(steps)
        for step, (x, y) in enumerate(dataloader):
            x, y = cast(Tensor, x).to(device), cast(Tensor, y).to(device)
            with context:
                logits = cast(Tensor, model(x, all_tokens=True))
                batch_size, seq_len, vocab_size = logits.shape
                loss = F.cross_entropy(logits.view(batch_size * seq_len, vocab_size), y.view(-1), ignore_index=-1)
            losses[step] = loss
            if step >= steps - 1:
                break
        all_losses.append(losses.mean().item())
    model.train()
    return all_losses

def unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, '_orig_mod') or hasattr(model, 'module'):
        if hasattr(model, '_orig_mod'):
            model = cast(nn.Module, model._orig_mod)
        elif hasattr(model, 'module'):
            model = cast(nn.Module, model.module)
    return model

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler, 
    scaler: torch.GradScaler,
    args: TrainArgs,
    step_number: int,
    val_loss: float,
    best_val_loss: float,
    checkpoints_folder_path: str
) -> None:
    checkpoint = Checkpoint(
        model=unwrap_model(model=model).state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(),
        args=args,
        step_number=step_number,
        val_loss=val_loss
    )
    torch.save(checkpoint, os.path.join(checkpoints_folder_path, f'{args.model_name}-{step_number}.pt'))
    if val_loss < best_val_loss:
        torch.save(checkpoint, os.path.join(checkpoints_folder_path, f'{args.model_name}-best.pt'))

def parse_train_args(
    model_names: list[str],
    optimizer_names: list[str],
    scheduler_names: list[str],
    dataset_names: list[str],
    default_optimizer_name: str,
    default_scheduler_name: str
) -> TrainArgs:
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--model_name', type=str, choices=model_names, required=True)
    args_parser.add_argument('--dataset_name', type=str, choices=dataset_names, required=True)
    args_parser.add_argument('--batch_size', type=int, default=16)
    args_parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    args_parser.add_argument('--grad_clip', type=int, default=1.0)
    args_parser.add_argument('--vocab_size', type=int, default=50304)
    args_parser.add_argument('--seq_len', type=int, default=128)
    args_parser.add_argument('--embed_dim', type=int, default=384)
    args_parser.add_argument('--hidden_dim', type=int, default=384*4)
    args_parser.add_argument('--num_transformer_blocks', type=int, default=4)
    args_parser.add_argument('--num_heads', type=int, default=4)
    args_parser.add_argument('--dropout_value', type=float, default=0.0)
    args_parser.add_argument('--use_bias', type=bool, default=False)
    args_parser.add_argument('--use_flash_attn', type=bool, default=True)
    args_parser.add_argument('--optimizer_name', type=str, choices=optimizer_names, default=default_optimizer_name)
    args_parser.add_argument('--scheduler_name', type=str, choices=scheduler_names, default=default_scheduler_name)
    args_parser.add_argument('--scheduler_max_steps', type=int, default=2000)
    args_parser.add_argument('--lr', type=float, default=6e-4)
    args_parser.add_argument('--lr_min', type=float, default=6e-5)
    args_parser.add_argument('--weight_decay', type=float, default=0.1)
    args_parser.add_argument('--beta1', type=float, default=0.9)
    args_parser.add_argument('--beta2', type=float, default=0.95)
    args_parser.add_argument('--train_steps', type=int, default=2000)
    args_parser.add_argument('--eval_steps', type=int, default=100)
    args_parser.add_argument('--eval_interval', type=int, default=100)
    args_parser.add_argument('--checkpoint_interval', type=int, default=100)
    args_parser.add_argument('--compile', type=bool, default=False)
    args_parser.add_argument('--use_cuda', type=bool, default=True)
    args = args_parser.parse_args()
    return TrainArgs(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        grad_clip=args.grad_clip,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_transformer_blocks=args.num_transformer_blocks,
        num_heads=args.num_heads,
        dropout_value=args.dropout_value,
        use_bias=args.use_bias,
        use_flash_attn=args.use_flash_attn,
        optimizer_name=args.optimizer_name,
        scheduler_name=args.scheduler_name,
        scheduler_max_steps=args.scheduler_max_steps,
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        train_steps=args.train_steps,
        eval_steps=args.eval_steps,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        compile=args.compile,
        use_cuda=args.use_cuda
    )

def parse_sample_args() -> SampleArgs:
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--checkpoint_file_name', type=str, required=True)
    args_parser.add_argument('--prompt', type=str, required=True)
    args_parser.add_argument('--temperature', type=float, default=0.8)
    args_parser.add_argument('--top_k', type=int, default=None)
    args_parser.add_argument('--max_new_tokens', type=int, default=200)
    args_parser.add_argument('--compile', type=bool, default=False)
    args_parser.add_argument('--use_cuda', type=bool, default=True)
    args = args_parser.parse_args()
    return SampleArgs(
        checkpoint_file_name=args.checkpoint_file_name,
        prompt=args.prompt,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        compile=args.compile,
        use_cuda=args.use_cuda
    )
