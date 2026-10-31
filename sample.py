"""
Generate text from a trained checkpoint.

Usage (from the project folder):
    python sample.py                                   # newest checkpoint in log/, default prompt
    python sample.py --prompt "The meaning of life is" --num_samples 3 --max_tokens 100
    python sample.py --ckpt log/model_28609.pt --temperature 0.8
"""
import argparse
import glob
import os

import tiktoken
import torch
from torch.nn import functional as F

# GPTConfig must be importable here: the checkpoint stores the config object,
# which was pickled as __main__.GPTConfig when train_gpt2.py saved it.
from model import GPT, GPTConfig  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=None, help="checkpoint path (default: newest log/model_*.pt)")
parser.add_argument("--prompt", default="Hello, I'm a language model,")
parser.add_argument("--num_samples", type=int, default=4)
parser.add_argument("--max_tokens", type=int, default=64, help="new tokens to generate")
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# find the checkpoint
ckpt_path = args.ckpt or sorted(glob.glob(os.path.join("log", "model_*.pt")))[-1]
print(f"loading {ckpt_path} on {device}")
# weights_only=False: the checkpoint contains a GPTConfig object, not just tensors
# (only do this with checkpoints you created yourself)
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
print(f"step {checkpoint['step']}, val loss {checkpoint['val_loss']:.4f}")

# rebuild the model and load the trained weights
model = GPT(checkpoint["config"])
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

enc = tiktoken.get_encoding("gpt2")
tokens = torch.tensor(enc.encode(args.prompt), dtype=torch.long)
x = tokens.unsqueeze(0).repeat(args.num_samples, 1).to(device)  # (num_samples, T)

rng = torch.Generator(device=device)
rng.manual_seed(args.seed)
block_size = model.config.block_size

with torch.no_grad():
    for _ in range(args.max_tokens):
        x_cond = x[:, -block_size:]  # crop to the context window
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits, _ = model(x_cond)
        logits = logits[:, -1, :].float()  # (B, vocab_size)
        logits = logits[:, :enc.n_vocab]   # ignore the padding tokens (vocab was padded to 50304)
        logits = logits / args.temperature
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, args.top_k, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=rng)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(args.num_samples):
    print(f"\n--- sample {i} ---")
    print(enc.decode(x[i].tolist()))