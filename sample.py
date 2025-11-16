import os
import tiktoken
import torch
from contextlib import nullcontext
from typing import cast

from common.types import Checkpoint
from common.constants import (
    SEED,
    MODEL_MAPPING,
    MODEL_CONFIG_MAPPING,
    DATASET_TOKEN_ENCODING_MAPPING,
    TOKEN_ENCODING_SPECIAL_TOKENS,
    CHECKPOINTS_FOLDER_PATH
)
from common.helpers import parse_sample_args

if __name__ == '__main__':
    args = parse_sample_args()

    device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if args.use_cuda and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    context = torch.autocast(device_type=device.type, dtype=dtype) if device.type == 'cuda' else nullcontext()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoint: Checkpoint = torch.load(
        os.path.join(CHECKPOINTS_FOLDER_PATH, args.checkpoint_file_name),
        map_location=device,
        weights_only=False
    )

    ModelType = MODEL_MAPPING[checkpoint.args.model_name]
    ModelConfigType = MODEL_CONFIG_MAPPING[checkpoint.args.model_name]
    model = ModelType(
        config=ModelConfigType(
            vocab_size=checkpoint.args.vocab_size,
            seq_len=checkpoint.args.seq_len,
            embed_dim=checkpoint.args.embed_dim,
            hidden_dim=checkpoint.args.hidden_dim,
            num_transformer_blocks=checkpoint.args.num_transformer_blocks,
            num_heads=checkpoint.args.num_heads,
            dropout_value=checkpoint.args.dropout_value,
            use_bias=checkpoint.args.use_bias,
            use_flash_attn=checkpoint.args.use_flash_attn
        )
    ).to(device)
    if args.compile:
        model = cast(ModelType, torch.compile(model))
    model.eval()

    token_encoding = DATASET_TOKEN_ENCODING_MAPPING[checkpoint.args.dataset_name]
    tokenizer = tiktoken.get_encoding(token_encoding.value)
    tokenized_prompt = tokenizer.encode(args.prompt, allowed_special=TOKEN_ENCODING_SPECIAL_TOKENS[token_encoding])
    
    x = torch.tensor([tokenized_prompt], dtype=torch.long).to(device)
    with torch.no_grad():
        with context:
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
            print(tokenizer.decode(y[0].tolist()))
