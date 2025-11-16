import os
import time
import math
import torch
import torch.nn.functional as F
from torch import Tensor
from contextlib import nullcontext
from typing import cast
from glob import glob

from common.constants import (
    SEED,
    MODEL_MAPPING,
    MODEL_CONFIG_MAPPING,
    OPTIMIZER_MAPPING,
    SCHEDULER_MAPPING,
    DEFAULT_OPTIMIZER,
    DEFAULT_SCHEDULER,
    DATA_FOLDER_PATH,
    TRAIN_SPLIT_NAME,
    VAL_SPLIT_NAME,
    CHECKPOINTS_FOLDER_PATH
)
from common.helpers import (
    parse_train_args,
    build_split_dataloader,
    build_optimizer,
    compute_loss,
    get_next_batch,
    save_checkpoint
)

if __name__ == '__main__':
    dataset_names = [os.path.basename(dataset_folder_path) for dataset_folder_path in glob(os.path.join(DATA_FOLDER_PATH, '*')) if os.path.isdir(dataset_folder_path)]
    args = parse_train_args(
        model_names=[model_iter.value for model_iter in list(MODEL_MAPPING.keys())],
        optimizer_names=[optimizer_iter.value for optimizer_iter in list(OPTIMIZER_MAPPING.keys())],
        scheduler_names=[scheduler_iter.value for scheduler_iter in list(SCHEDULER_MAPPING.keys())],
        default_optimizer_name=DEFAULT_OPTIMIZER.value,
        default_scheduler_name=DEFAULT_SCHEDULER.value,
        dataset_names=dataset_names
    )

    device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if args.use_cuda and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    context = torch.autocast(device_type=device.type, dtype=dtype) if device.type == 'cuda' else nullcontext()

    train_dataloader = build_split_dataloader(
        file_path=os.path.join(DATA_FOLDER_PATH, args.dataset_name, f'{TRAIN_SPLIT_NAME}.npy'),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )
    val_dataloader = build_split_dataloader(
        file_path=os.path.join(DATA_FOLDER_PATH, args.dataset_name, f'{VAL_SPLIT_NAME}.npy'),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    assert len(train_dataloader) > 0, 'Empty train dataloader'
    assert len(val_dataloader) > 0, 'Empty val dataloader'

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ModelType = MODEL_MAPPING[args.model_name]
    ModelConfigType = MODEL_CONFIG_MAPPING[args.model_name]
    model = ModelType(
        config=ModelConfigType(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_transformer_blocks=args.num_transformer_blocks,
            num_heads=args.num_heads,
            dropout_value=args.dropout_value,
            use_bias=args.use_bias,
            use_flash_attn=args.use_flash_attn
        )
    ).to(device)
    if args.compile:
        model = cast(ModelType, torch.compile(model))

    optimizer, scheduler = build_optimizer(
        model=model,
        device=device,
        optimizer_type=OPTIMIZER_MAPPING[args.optimizer_name],
        scheduler_type=SCHEDULER_MAPPING[args.scheduler_name],
        lr=args.lr,
        lr_min=args.lr_min,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        scheduler_max_steps=args.scheduler_max_steps
    )
    scaler = torch.GradScaler(device=device.type, enabled=dtype==torch.float16)

    for file_path in glob(os.path.join(CHECKPOINTS_FOLDER_PATH, f'{args.model_name}-*.pt')):
        if os.path.isfile(file_path):
            os.remove(file_path)

    step_number = 0
    best_val_loss: float = math.inf
    start_time = time.time()
    train_dataloader_iter = iter(train_dataloader)
    for step_number in range(1, args.train_steps + 1):
        for gradient_accumulation_step in range(args.gradient_accumulation_steps):
            (train_x, train_y), train_dataloader_iter = get_next_batch(dataloader=train_dataloader, dataloader_iter=train_dataloader_iter, device=device)
            with context:
                logits = cast(Tensor, model(train_x))
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), train_y.view(-1), ignore_index=-1)
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

        if args.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        perform_eval = step_number % args.eval_interval == 0
        perform_checkpoint = step_number % args.checkpoint_interval == 0

        if perform_eval or perform_checkpoint:
            train_loss, val_loss = compute_loss(
                model=model,
                dataloaders=[train_dataloader, val_dataloader],
                steps=args.eval_steps,
                context=context,
                device=device
            )

            elapsed_time = time.time() - start_time
            elapsed_hours = int(elapsed_time // 3600)
            elapsed_minutes = int((elapsed_time % 3600) // 60)
            print(f'{elapsed_hours:02d}:{elapsed_minutes:02d} Step {step_number}: train_loss {train_loss} - val_loss {val_loss}')

            if perform_checkpoint:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    step_number=step_number,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    checkpoints_folder_path=CHECKPOINTS_FOLDER_PATH
                )
                print('Checkpoint')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
    train_loss, val_loss = compute_loss(
        model=model,
        dataloaders=[train_dataloader, val_dataloader],
        steps=args.eval_steps,
        context=context,
        device=device
    )
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
        step_number=step_number,
        val_loss=val_loss,
        best_val_loss=best_val_loss,
        checkpoints_folder_path=CHECKPOINTS_FOLDER_PATH
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
