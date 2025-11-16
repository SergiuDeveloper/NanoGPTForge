import sys
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from dataclasses import dataclass
from typing import cast, Any

class TransformerMLP(nn.Module):
    @dataclass
    class Config:
        embed_dim: int
        hidden_dim: int
        dropout_value: float
        use_bias: bool

    def __init__(self, config: Config):
        super().__init__()

        self.fc = nn.Linear(config.embed_dim, config.hidden_dim, bias=config.use_bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.hidden_dim, config.embed_dim, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout_value)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    @dataclass
    class Config:
        embed_dim: int
        hidden_dim: int
        num_heads: int
        dropout_value: float
        use_bias: bool
        use_flash_attn: bool

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        self.ln_1 = nn.LayerNorm(config.embed_dim, bias=config.use_bias)
        self.attn = nn.MultiheadAttention(config.embed_dim, config.num_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(config.embed_dim, bias=config.use_bias)
        self.mlp = TransformerMLP(config=TransformerMLP.Config(
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            dropout_value=config.dropout_value,
            use_bias=config.use_bias
        ))

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.ln_1(x)
        if self.config.use_flash_attn and not x.is_cuda:
            print('Warning: Attempted to use FlashAttention without CUDA')
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if self.config.use_flash_attn and x.is_cuda else sdpa_kernel(SDPBackend.MATH):
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x
    
class NanoGPT(nn.Module):
    @dataclass
    class Config:
        vocab_size: int
        seq_len: int
        embed_dim: int
        hidden_dim: int
        num_transformer_blocks: int
        num_heads: int
        dropout_value: float
        use_bias: bool
        use_flash_attn: bool

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.embed_dim),
            wpe=nn.Embedding(config.seq_len, config.embed_dim),
            dropout=nn.Dropout(config.dropout_value),
            h=nn.ModuleList([TransformerBlock(config=TransformerBlock.Config(
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout_value=config.dropout_value,
                use_bias=config.use_bias,
                use_flash_attn=config.use_flash_attn
            )) for _ in range(config.num_transformer_blocks)]),
            ln_f=nn.LayerNorm(config.embed_dim, bias=config.use_bias),
        ))
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=config.use_bias)

        self.lm_head.weight = cast(Tensor, cast(nn.Module, self.transformer.wte).weight)

        self.apply(self.__init_weights)
        for transformer_block in cast(nn.ModuleList, self.transformer.h).children():
            c_proj = cast(TransformerBlock, transformer_block).mlp.c_proj
            c_proj.weight = torch.nn.init.normal_(c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.num_transformer_blocks))

    def forward(self, x: Tensor, all_tokens: bool=False) -> Tensor:
        device = x.device
        _, seq_len = x.size()
        if seq_len > self.config.seq_len:
            print(f'Warning: Attempted forward with sequence length {seq_len} longer than context length {self.seq_len}', file=sys.stderr)
        x = x[:, -self.config.seq_len:]

        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        tok_emb = cast(nn.Module, self.transformer.wte)(x)
        pos_emb = cast(nn.Module, self.transformer.wpe)(pos)
        pos_emb = pos_emb.unsqueeze(0)
        x = cast(nn.Module, self.transformer.dropout)(tok_emb + pos_emb)
        for transformer_block in cast(nn.ModuleList, self.transformer.h):
            x = transformer_block(x)
        x = cast(nn.Module, self.transformer.ln_f)(x)

        if self.training or all_tokens:
            return self.lm_head(x)
        return self.lm_head(x[:, [-1], :])

    @torch.no_grad()
    def generate(self, x: Tensor, new_tokens_count: int, temperature: float, top_k: int | None=None) -> Tensor:
        for _ in range(new_tokens_count):
            x_trunc = x if x.size(1) <= self.config.seq_len else x[:, -self.config.seq_len:]
            logits = self.forward(x_trunc, all_tokens=False)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                topk_vals, _ = torch.topk(logits, top_k, dim=-1)
                min_topk = topk_vals[:, -1].unsqueeze(-1)
                logits[logits < min_topk] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            y = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, y), dim=1)
        return x

    def __init_weights(self, module: nn.Linear | nn.Embedding | Any) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            return
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
