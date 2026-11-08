"""Train the tiny Vision Mamba used by the explainer and export its weights.

The JavaScript forward pass in src/lib/vim.js follows this file. Shapes below are
the contract: change both sides together.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

D = 16
INNER = 32
N = 4
CONV = 3
DT_RANK = 4
LAYERS = 4
CLASSES = ["Circle", "Square", "Stripes", "Diagonal"]
IMG = 24
PATCH = 8
GRID = IMG // PATCH
N_PATCH = GRID * GRID
CLS_AT = N_PATCH // 2
L = N_PATCH + 1
ROOT = Path(__file__).resolve().parent


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def causal_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv. weight[:, 0] sees the oldest tap."""
    acc = bias.view(1, 1, -1) + torch.zeros_like(x)
    k = weight.shape[1]
    for tap in range(k):
        shift = k - 1 - tap
        kernel = weight[:, tap].view(1, 1, -1)
        if shift == 0:
            acc = acc + x * kernel
        else:
            acc[:, shift:, :] = acc[:, shift:, :] + x[:, :-shift, :] * kernel
    return acc


def scan(x, dt, b, c, a, d_skip):
    """Selective scan. x (B,L,I), dt (B,L,I), b/c (B,L,N), a (I,N)."""
    hidden = x.new_zeros(x.shape[0], x.shape[2], a.shape[1])
    outputs = []
    for t in range(x.shape[1]):
        a_bar = torch.exp(dt[:, t].unsqueeze(-1) * a)
        b_bar = dt[:, t].unsqueeze(-1) * b[:, t].unsqueeze(1)
        hidden = a_bar * hidden + b_bar * x[:, t].unsqueeze(-1)
        y = (hidden * c[:, t].unsqueeze(1)).sum(-1) + d_skip * x[:, t]
        outputs.append(y)
    return torch.stack(outputs, dim=1)


class Direction(nn.Module):
    """One selective scan. Vision Mamba keeps a separate one for each way."""

    def __init__(self):
        super().__init__()
        self.conv_w = nn.Parameter(torch.randn(INNER, CONV) * 0.05)
        self.conv_b = nn.Parameter(torch.zeros(INNER))
        self.x_proj = nn.Linear(INNER, DT_RANK + 2 * N, bias=False)
        self.dt_proj = nn.Linear(DT_RANK, INNER)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, N + 1, dtype=torch.float32)).repeat(INNER, 1))
        self.D = nn.Parameter(torch.ones(INNER))
        dt = torch.exp(torch.rand(INNER) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv)
            self.dt_proj.weight.normal_(0, 0.01)

    def forward(self, x):
        activated = silu(causal_conv(x, self.conv_w, self.conv_b))
        projected = self.x_proj(activated)
        delta, b, c = torch.split(projected, [DT_RANK, N, N], dim=-1)
        step = F.softplus(self.dt_proj(delta))
        y = scan(activated, step, b, c, -torch.exp(self.A_log), self.D)
        return y


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.in_proj = nn.Linear(D, 2 * INNER)
        self.forward_scan = Direction()
        self.backward_scan = Direction()
        self.out_proj = nn.Linear(INNER, D)

    def forward(self, hidden):
        residual = hidden
        normed = self.norm(hidden)
        x, z = self.in_proj(normed).chunk(2, dim=-1)
        forward = self.forward_scan(x)
        backward = self.backward_scan(x.flip(1)).flip(1)
        gated = (forward + backward) * silu(z)
        return residual + self.out_proj(gated)


class TinyVim(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Linear(3 * PATCH * PATCH, D)
        self.cls = nn.Parameter(torch.zeros(D))
        self.pos = nn.Parameter(torch.zeros(L, D))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, len(CLASSES))

    def forward(self, images):
        tokens = self.embed(images)
        for block in self.blocks:
            tokens = block(tokens)
        cls = self.norm(tokens)[:, CLS_AT, :]
        return self.head(cls)

    def embed(self, images):
        patches = []
        for row in range(GRID):
            for col in range(GRID):
                crop = images[
                    :,
                    :,
                    row * PATCH : (row + 1) * PATCH,
                    col * PATCH : (col + 1) * PATCH,
                ]
                patches.append(crop.reshape(images.shape[0], -1))
        tokens = self.patch(torch.stack(patches, dim=1))
        cls = self.cls.view(1, 1, D).expand(images.shape[0], 1, D)
        sequence = torch.cat([tokens[:, :CLS_AT], cls, tokens[:, CLS_AT:]], dim=1)
        return sequence + self.pos


def make_image(kind: int, rng: random.Random, noise: float = 0.05) -> torch.Tensor:
    image = torch.rand(3, IMG, IMG) * 0.08
    color = torch.tensor(
        [rng.uniform(0.35, 1), rng.uniform(0.35, 1), rng.uniform(0.35, 1)]
    ).view(3, 1, 1)
    axis = torch.linspace(-1, 1, IMG)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    if kind == 0:
        cy, cx = rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)
        rad = rng.uniform(0.35, 0.72)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad**2
    elif kind == 1:
        cy, cx = rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)
        sy, sx = rng.uniform(0.32, 0.68), rng.uniform(0.32, 0.68)
        mask = ((yy - cy).abs() <= sy) & ((xx - cx).abs() <= sx)
    elif kind == 2:
        period = rng.choice([4, 5, 6])
        phase = rng.randrange(period)
        rows = (torch.arange(IMG) + phase) % (period * 2) < period
        mask = rows.view(IMG, 1).expand(IMG, IMG)
    else:
        shift = rng.uniform(-0.25, 0.25)
        width = rng.uniform(0.12, 0.28)
        mask = (yy - xx - shift).abs() <= width
    image = torch.where(mask.unsqueeze(0), color.expand_as(image), image)
    if noise:
        image = image + noise * torch.randn_like(image)
    return image.clamp(0, 1)


def canonical(kind: int) -> torch.Tensor:
    image = torch.full((3, IMG, IMG), 0.05)
    axis = torch.linspace(-1, 1, IMG)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    if kind == 0:
        color = torch.tensor([0.93, 0.33, 0.28]).view(3, 1, 1)
        mask = yy**2 + xx**2 <= 0.55**2
    elif kind == 1:
        color = torch.tensor([0.27, 0.47, 0.95]).view(3, 1, 1)
        mask = (yy.abs() <= 0.55) & (xx.abs() <= 0.55)
    elif kind == 2:
        color = torch.tensor([0.18, 0.72, 0.42]).view(3, 1, 1)
        rows = (torch.arange(IMG) % 6) < 3
        mask = rows.view(IMG, 1).expand(IMG, IMG)
    else:
        color = torch.tensor([0.95, 0.76, 0.22]).view(3, 1, 1)
        mask = (yy - xx).abs() <= 0.18
    return torch.where(mask.unsqueeze(0), color, image)


def batch(rng: random.Random, size: int, clean: float = 0.25):
    kinds = [rng.randrange(len(CLASSES)) for _ in range(size)]
    images = []
    for kind in kinds:
        if rng.random() < clean:
            images.append(make_image(kind, rng, noise=0.02))
        else:
            images.append(make_image(kind, rng))
    return torch.stack(images), torch.tensor(kinds)


def evaluate(model, device, n: int = 240) -> float:
    rng = random.Random(1234)
    images, kinds = batch(rng, n, clean=0.5)
    with torch.no_grad():
        pred = model(images.to(device)).argmax(-1).cpu()
    return float((pred == kinds).float().mean())


def dump_direction(direction):
    return {
        "conv_w": direction.conv_w.detach().cpu().tolist(),
        "conv_b": direction.conv_b.detach().cpu().tolist(),
        "x_weight": direction.x_proj.weight.detach().cpu().tolist(),
        "dt_weight": direction.dt_proj.weight.detach().cpu().tolist(),
        "dt_bias": direction.dt_proj.bias.detach().cpu().tolist(),
        "A": (-torch.exp(direction.A_log)).detach().cpu().tolist(),
        "D": direction.D.detach().cpu().tolist(),
    }


def export(model):
    model.eval()
    blocks = []
    for block in model.blocks:
        blocks.append(
            {
                "ln_weight": block.norm.weight.detach().cpu().tolist(),
                "ln_bias": block.norm.bias.detach().cpu().tolist(),
                "in_weight": block.in_proj.weight.detach().cpu().tolist(),
                "in_bias": block.in_proj.bias.detach().cpu().tolist(),
                "forward": dump_direction(block.forward_scan),
                "backward": dump_direction(block.backward_scan),
                "out_weight": block.out_proj.weight.detach().cpu().tolist(),
                "out_bias": block.out_proj.bias.detach().cpu().tolist(),
            }
        )
    weights = {
        "meta": {
            "d": D,
            "inner": INNER,
            "state": N,
            "conv": CONV,
            "dtRank": DT_RANK,
            "layers": LAYERS,
            "classes": CLASSES,
            "image": IMG,
            "patch": PATCH,
            "clsAt": CLS_AT,
            "length": L,
        },
        "patch_weight": model.patch.weight.detach().cpu().tolist(),
        "patch_bias": model.patch.bias.detach().cpu().tolist(),
        "cls": model.cls.detach().cpu().tolist(),
        "pos": model.pos.detach().cpu().tolist(),
        "blocks": blocks,
        "final_ln_weight": model.norm.weight.detach().cpu().tolist(),
        "final_ln_bias": model.norm.bias.detach().cpu().tolist(),
        "head_weight": model.head.weight.detach().cpu().tolist(),
        "head_bias": model.head.bias.detach().cpu().tolist(),
    }
    out = ROOT / "src" / "lib" / "weights.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(weights))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


def export_examples(model, device):
    examples = []
    checks = []
    for kind, name in enumerate(CLASSES):
        image = canonical(kind)
        with torch.no_grad():
            logits = model(image[None].to(device))[0].cpu()
        pred = int(logits.argmax())
        print(f"canonical {name}: pred={CLASSES[pred]} logits={[round(v, 3) for v in logits.tolist()]}")
        packed = image.tolist()
        examples.append({"id": name.lower(), "label": name, "classIndex": kind, "image": packed})
        if kind == 0:
            checks.append({"image": packed, "logits": logits.tolist()})
    lib = ROOT / "src" / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    (lib / "examples.json").write_text(json.dumps({"examples": examples}))
    (lib / "check.json").write_text(json.dumps(checks[0]))


def main():
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyVim().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = random.Random(0)
    best_acc = -1.0
    best_state = None
    for step in range(1, 1601):
        model.train()
        images, kinds = batch(rng, 64)
        logits = model(images.to(device))
        loss = F.cross_entropy(logits, kinds.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            model.eval()
            acc = evaluate(model, device)
            print(f"step {step:4d}  loss {loss.item():.3f}  acc {acc:.3f}")
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if acc >= 0.98 and step >= 400:
                break
    model.load_state_dict(best_state)
    model.to(device).eval()
    print(f"best acc {best_acc:.3f}")
    export(model)
    export_examples(model, device)


if __name__ == "__main__":
    main()
