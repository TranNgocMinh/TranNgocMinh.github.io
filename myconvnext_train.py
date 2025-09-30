
import argparse
import math
import os
import random
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision.transforms as T

# -----------------------------
# Import model
# -----------------------------
from myconvnext import MyConvNeXtTiny

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def linear_lr_scale(lr: float, batch_size: int, accum_steps: int, base_bs: int = 256) -> float:
    eff = batch_size * max(1, accum_steps)
    return lr * eff / base_bs


# -----------------------------
# Dataset (FER2013 CSV)
# -----------------------------
USAGE_ALIASES = {
    "train": ["training", "train", "training set", "publictrain", "public train"],
    "val":   ["publictest", "public test", "val", "validation"],
    "test":  ["privatetest", "private test", "test"],
}

def clean_fer_df(df: pd.DataFrame) -> pd.DataFrame:
    def ok(p):
        s = re.sub(r"[^0-9]+", " ", str(p)).strip()
        return len(s.split()) == 48 * 48
    mask = df["pixels"].map(ok)
    bad = (~mask).sum()
    if bad:
        print(f"[warn] Dropping {bad} corrupted rows (invalid pixels length).")
    return df[mask].reset_index(drop=True)

def _parse_pixels_to_pil(pixels: str, img_size: int) -> Image.Image:
    s = re.sub(r"[^0-9]+", " ", str(pixels)).strip()
    toks = s.split()
    if len(toks) != 48 * 48:
        raise ValueError(f"pixels length={len(toks)} != 2304. Check CSV 'pixels' column")
    arr = np.array([int(t) for t in toks], dtype=np.uint8).reshape(48, 48)
    pil = Image.fromarray(arr).convert("L")
    if img_size != 48:
        pil = pil.resize((img_size, img_size), resample=Image.BILINEAR)
    return pil

class FER2013CSV(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int = 224, augment: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        if augment:
            self.transform = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandAugment(num_ops=2, magnitude=9),
                T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(int(img_size * 1.14)),
                T.CenterCrop(img_size),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = _parse_pixels_to_pil(row["pixels"], self.img_size)
        x = self.transform(img)
        y = int(row["emotion"])
        return x, y


# -----------------------------
# Data split helper
# -----------------------------
def split_by_usage(df: pd.DataFrame, seed: int = 42, val_ratio: float = 0.1, test_ratio: float = 0.1,
                   force_random_split: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if (not force_random_split) and ("Usage" in df.columns):
        usage = df["Usage"].astype(str).str.strip().str.lower()
        def map_usage(u: str) -> str:
            if u in USAGE_ALIASES["train"]: return "train"
            if u in USAGE_ALIASES["val"]:   return "val"
            if u in USAGE_ALIASES["test"]:  return "test"
            if u == "training": return "train"
            if u == "publictest": return "val"
            if u == "privatetest": return "test"
            return u
        u = usage.map(map_usage)
        df["Usage_norm"] = u
        tr = df[df["Usage_norm"] == "train"].copy()
        va = df[df["Usage_norm"] == "val"].copy()
        te = df[df["Usage_norm"] == "test"].copy()
        if len(va) == 0 or len(te) == 0 or len(tr) == 0:
            rng = np.random.default_rng(seed)
            idx = df.index.to_numpy()
            rng.shuffle(idx)
            n_val = max(1, int(len(idx) * val_ratio))
            te_idx = idx[:max(1, int(len(idx) * test_ratio))]
            va_idx = idx[max(1, int(len(idx) * test_ratio)):max(1, int(len(idx) * test_ratio))+n_val]
            tr_idx = idx[max(1, int(len(idx) * test_ratio))+n_val:]
            tr = df.loc[tr_idx].copy()
            va = df.loc[va_idx].copy()
            te = df.loc[te_idx].copy()
        return tr, va, te
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = max(1, int(len(df) * val_ratio))
    n_te = max(1, int(len(df) * test_ratio))
    va_idx = idx[:n_val]
    te_idx = idx[n_val:n_val+n_te]
    tr_idx = idx[n_val+n_te:]
    return df.iloc[tr_idx].copy(), df.iloc[va_idx].copy(), df.iloc[te_idx].copy()


# -----------------------------
# Losses & mix
# -----------------------------
class LabelSmoothingCE(nn.Module):
    def __init__(self, classes: int, eps: float = 0.1):
        super().__init__()
        self.eps = eps
        self.classes = classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=1)
        n = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logp)
            true_dist.fill_(self.eps / (n - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.eps)
        return torch.mean(torch.sum(-true_dist * logp, dim=1))


def soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=1)
    return torch.mean(torch.sum(-soft_targets * logp, dim=1))


def rand_bbox(size, lam):
    B, C, H, W = size
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def apply_mix(x, y, num_classes: int, mixup_alpha: float, cutmix_alpha: float, enable_mix: bool):
    if not enable_mix or (mixup_alpha <= 0 and cutmix_alpha <= 0):
        return x, y, None, 1.0
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < 0.5)
    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        perm = torch.randperm(x.size(0), device=x.device)
        x2, y2 = x[perm], y[perm]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bby1:bby2, bbx1:bbx2] = x2[:, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        perm = torch.randperm(x.size(0), device=x.device)
        x2, y2 = x[perm], y[perm]
        x = x * lam + x2 * (1 - lam)
    # build soft targets
    y1 = torch.zeros(x.size(0), num_classes, device=x.device)
    y1.scatter_(1, y.unsqueeze(1), 1.0)
    y2_onehot = torch.zeros_like(y1)
    y2_onehot.scatter_(1, y2.unsqueeze(1), 1.0)
    soft = y1 * lam + y2_onehot * (1 - lam)
    return x, None, soft, lam


# -----------------------------
# EMA helper
# -----------------------------
class ModelEma:
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        self.ema = self._clone(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema.to(device)

    @torch.no_grad()
    def _clone(self, model):
        ema = MyConvNeXtTiny(in_chans=3, num_classes=model.head.out_features, drop_path_rate=0.0)
        ema.load_state_dict(model.state_dict(), strict=True)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].to(v.device), alpha=1 - d)
            else:
                self.ema.state_dict()[k].copy_(msd[k])


# -----------------------------
# Scheduler (warmup + cosine)
# -----------------------------
def build_warmup_cosine(optimizer, epochs: int, warmup_epochs: int, min_lr: float):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = (current_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return cosine
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# -----------------------------
# Evaluation (with optional TTA)
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device, tta: bool = False) -> Tuple[float, float]:
    model.eval()
    total, correct = 0, 0
    total_loss = 0.0
    ce = nn.CrossEntropyLoss(reduction='sum')
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if tta:
            logits1 = model(x)
            logits2 = model(torch.flip(x, dims=[3]))
            logits = (logits1 + logits2) / 2
        else:
            logits = model(x)
        loss = ce(logits, y)
        total_loss += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = correct / max(1, total)
    avg_loss = total_loss / max(1, total)
    return acc, avg_loss


# -----------------------------
# Main train loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # Data & IO
    parser.add_argument('--csv', type=str, required=True, help='Path to FER2013 CSV file')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--balanced-sampler', action='store_true')
    parser.add_argument('--force-random-split', action='store_true')
    # Optimization
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--accum-steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--no-lr-scale', action='store_true')
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=5e-2)
    parser.add_argument('--drop-path', type=float, default=0.2)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    # Regularization / mix
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--mixup', type=float, default=0.0, help='alpha for MixUp (0 = off)')
    parser.add_argument('--cutmix', type=float, default=0.0, help='alpha for CutMix (0 = off)')
    parser.add_argument('--no-mix', action='store_true', help='disable MixUp/CutMix quickly')
    # Runtime
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--ema-decay', type=float, default=0.9997)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--compile', action='store_true', help='torch.compile for speed (PyTorch 2+)')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--out', type=str, default='best_myconvnext.pth')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience (epochs)')
    parser.add_argument('--tta', action='store_true')

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load & clean CSV
    df = pd.read_csv(args.csv, dtype={"pixels": str})
    df = clean_fer_df(df)
    train_df, val_df, _ = split_by_usage(df, seed=args.seed, force_random_split=args.force_random_split)

    # Datasets & loaders
    tr_ds = FER2013CSV(train_df, img_size=args.img_size, augment=True)
    va_ds = FER2013CSV(val_df,   img_size=args.img_size, augment=False)

    if args.balanced_sampler:
        labels = train_df['emotion'].astype(int).to_numpy()
        class_counts = np.bincount(labels, minlength=7)
        class_weights = 1.0 / np.clip(class_counts, 1, None)
        weights = class_weights[labels]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler, shuffle=shuffle,
                           num_workers=args.workers, pin_memory=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=max(1, args.batch_size // 2), shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    # Model (Block mới bên trong MyConvNeXtTiny)
    model = MyConvNeXtTiny(in_chans=3, num_classes=7, drop_path_rate=args.drop_path)
    model.to(device)

    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}")

    base_lr = args.lr if args.no_lr_scale else linear_lr_scale(args.lr, args.batch_size, args.accum_steps)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine(optimizer, epochs=args.epochs, warmup_epochs=args.warmup, min_lr=args.min_lr)

    ce_smooth = LabelSmoothingCE(classes=7, eps=args.label_smoothing)

    ema = ModelEma(model, decay=args.ema_decay, device=device) if args.ema else None
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    start_epoch = 0
    best_acc = 0.0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt.get('model', ckpt))
        if 'optimizer' in ckpt:
            try: optimizer.load_state_dict(ckpt['optimizer'])
            except Exception: print('[warn] Optimizer state in resume is incompatible — reinit.')
        if 'scaler' in ckpt and args.amp:
            try: scaler.load_state_dict(ckpt['scaler'])
            except Exception: print('[warn] AMP scaler state in resume is incompatible — reinit.')
        if 'scheduler' in ckpt:
            try: scheduler.load_state_dict(ckpt['scheduler'])
            except Exception: print('[warn] Scheduler state in resume is incompatible — reinit.')
        start_epoch = ckpt.get('epoch', 0)
        best_acc = ckpt.get('best_acc', 0.0)
        if ema and 'ema' in ckpt and ckpt['ema'] is not None:
            try: ema.ema.load_state_dict(ckpt['ema'])
            except Exception: print('[warn] EMA state in resume is incompatible — reinit EMA.')
        print(f"[info] Resumed from {args.resume} @ epoch {start_epoch} (best_acc={best_acc:.4f})")

    epochs_no_improve = 0

    def save_ckpt(tag: str, epoch: int, best: float):
        path = os.path.join(args.checkpoint_dir, f"{tag}.pth")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict() if args.amp else None,
            'epoch': epoch,
            'best_acc': best,
            'ema': (ema.ema.state_dict() if ema else None),
            'args': vars(args),
        }, path)
        return path

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        optimizer.zero_grad(set_to_none=True)

        for it, (x, y) in enumerate(tr_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            enable_mix = (not args.no_mix) and (args.mixup > 0 or args.cutmix > 0)
            x_mixed, y_hard, y_soft, lam = apply_mix(
                x, y, num_classes=7, mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, enable_mix=enable_mix
            )

            with torch.amp.autocast('cuda', enabled=args.amp):
                logits = model(x_mixed)
                if y_soft is not None:
                    loss = soft_ce_loss(logits, y_soft)
                else:
                    loss = ce_smooth(logits, y)

            scaler.scale(loss / max(1, args.accum_steps)).backward()

            do_step = ((it + 1) % args.accum_steps == 0) or ((it + 1) == len(tr_loader))
            if do_step:
                if args.clip_grad and args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if ema:
                    ema.update(model)

            total_loss += loss.item() * x.size(0)
            with torch.no_grad():
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        scheduler.step()

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        eval_model = ema.ema if ema else model
        val_acc, val_loss = evaluate(eval_model, va_loader, device=device, tta=args.tta)

        print(f"Epoch {epoch+1}/{args.epochs} | LR {optimizer.param_groups[0]['lr']:.6f} | "
              f"train_loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
              f"val_loss {val_loss:.4f} acc {val_acc*100:.2f}%")

        save_ckpt('last', epoch+1, best_acc)
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            best_path = args.out if os.path.isabs(args.out) else os.path.join(args.checkpoint_dir, os.path.basename(args.out))
            torch.save(eval_model.state_dict(), best_path)
            print(f"[best] Saved best weights -> {best_path} (acc={best_acc*100:.2f}%)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"[early-stop] no improvement for {args.patience} epochs. Best acc={best_acc*100:.2f}%")
            break

    print("Done.")


if __name__ == '__main__':
    main()
