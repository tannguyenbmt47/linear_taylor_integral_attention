"""
Training script for DeiT Linear Taylor Integral on ImageNet-1K (Parquet format).

Usage:
    python train.py --config configs/deit_linear_taylor_integral.yaml
    python train.py --config configs/deit_linear_taylor_integral.yaml --data_dir /path/to/imagenet_parquet
    python train.py --config configs/deit_linear_taylor_integral.yaml --variant small --epochs 100
"""

import argparse
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from models.deit_linear_taylor_integral import (
    deit_tiny_linear_taylor_integral,
    deit_small_linear_taylor_integral,
    deit_base_linear_taylor_integral,
)
from models.deit_integral_attention import (
    deit_tiny_integral,
    deit_small_integral,
    deit_base_integral,
)
from models.deit_integral_diff import (
    deit_tiny_integral_diff,
    deit_small_integral_diff,
    deit_base_integral_diff,
)
from models.agent_deit import (
    deit_tiny_agent_attention,
    deit_small_agent_attention,
    deit_base_agent_attention,
)
from models.deit_token_agent import (
    DeiTTinyTokenAgent,
)
from utils.dataset import build_imagenet_dataset, build_hf_dataset, MixupCutmix


# ════════════════════════════════════════════════════════════
#  Config
# ════════════════════════════════════════════════════════════

def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_cli_overrides(cfg, args):
    """Override config values with CLI arguments when provided."""
    if args.data_dir:
        cfg["train"]["data_dir"] = args.data_dir
    if args.variant:
        cfg["model"]["variant"] = args.variant
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.output_dir:
        cfg["system"]["output_dir"] = args.output_dir
    if args.num_workers is not None:
        cfg["system"]["num_workers"] = args.num_workers
    if args.lazy_loading:
        cfg.setdefault("data", {})["lazy"] = True
    if args.resume:
        cfg["resume"] = args.resume
    return cfg


# ════════════════════════════════════════════════════════════
#  Model builder
# ════════════════════════════════════════════════════════════

MODEL_FACTORIES = {
    "deit_linear_taylor_integral": {
        "tiny": deit_tiny_linear_taylor_integral,
        "small": deit_small_linear_taylor_integral,
        "base": deit_base_linear_taylor_integral,
    },
    "deit_integral_attention": {
        "tiny": deit_tiny_integral,
        "small": deit_small_integral,
        "base": deit_base_integral,
    },
    "deit_integral_diff": {
        "tiny": deit_tiny_integral_diff,
        "small": deit_small_integral_diff,
        "base": deit_base_integral_diff,
    },
    "deit_agent_attention": {
        "tiny": deit_tiny_agent_attention,
        "small": deit_small_agent_attention,
        "base": deit_base_agent_attention,
    },
    "deit_tiny_token_agent": {
        "tiny": lambda **kw: DeiTTinyTokenAgent(
            img_size=kw.get('img_size', 224),
            patch_size=kw.get('patch_size', 16),
            in_chans=kw.get('in_channels', 3),
            num_classes=kw.get('num_classes', 1000),
            embed_dim=kw.get('embed_dim', 192),
            depth=kw.get('depth', 8),
            num_heads=kw.get('num_heads', 3),
            mlp_ratio=kw.get('mlp_ratio', 4.0),
            qkv_bias=kw.get('qkv_bias', True),
            drop_rate=kw.get('drop_rate', 0.3),
            attn_drop_rate=kw.get('attn_drop_rate', 0.2),
            drop_path_rate=kw.get('drop_path_rate', 0.4),
            agent_num=kw.get('agent_num', 36),
            d_k=kw.get('d_k', 64),
        ),
    },
}


def build_model(cfg):
    model_name = cfg["model"].get("name", "deit_linear_taylor_integral")
    variant = cfg["model"]["variant"]
    factories = MODEL_FACTORIES[model_name]
    factory = factories[variant]
    
    # Common model args
    model_args = {
        "num_classes": cfg["model"]["num_classes"],
        "img_size": cfg["model"]["img_size"],
        "patch_size": cfg["model"]["patch_size"],
        "in_channels": cfg["model"]["in_channels"],
        "dropout": cfg["model"].get("drop_rate", cfg["model"].get("dropout", 0.0)),
        "attn_dropout": cfg["model"].get("attn_drop_rate", cfg["model"].get("attn_dropout", 0.0)),
        "drop_path_rate": cfg["model"]["drop_path_rate"],
    }
    
    # Only pass use_dist_token if model supports it
    if model_name in ["deit_linear_taylor_integral", "deit_integral_attention", "deit_integral_diff"]:
        model_args["use_dist_token"] = cfg["model"].get("use_dist_token", False)
    
    # Add token agent specific args
    if model_name == "deit_tiny_token_agent":
        model_args.update({
            "embed_dim": cfg["model"].get("embed_dim", 192),
            "depth": cfg["model"].get("depth", 8),
            "num_heads": cfg["model"].get("num_heads", 3),
            "mlp_ratio": cfg["model"].get("mlp_ratio", 4.0),
            "qkv_bias": cfg["model"].get("qkv_bias", True),
            "drop_rate": cfg["model"].get("drop_rate", 0.3),
            "attn_drop_rate": cfg["model"].get("attn_drop_rate", 0.2),
            "agent_num": cfg["model"].get("agent_num", 36),
            "d_k": cfg["model"].get("d_k", 64),
        })
    
    model = factory(**model_args)
    return model


# ════════════════════════════════════════════════════════════
#  LR Scheduler (cosine with warmup)
# ════════════════════════════════════════════════════════════

def adjust_learning_rate(optimizer, epoch, cfg):
    """Cosine annealing with linear warmup."""
    warmup_epochs = cfg["train"]["warmup_epochs"]
    max_lr = cfg["train"]["lr"]
    min_lr = cfg["train"]["min_lr"]
    total_epochs = cfg["train"]["epochs"]

    if epoch < warmup_epochs:
        lr = max_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ════════════════════════════════════════════════════════════
#  Training & Validation
# ════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, mixup_fn, device, cfg, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    log_interval = cfg["system"]["log_interval"]
    use_dist = cfg["model"].get("use_dist_token", False)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Mixup / CutMix (produces soft labels)
        if mixup_fn is not None:
            images, soft_targets = mixup_fn(images, targets)
            use_soft = True
        else:
            use_soft = False

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            if use_dist:
                cls_out, dist_out = model(images)
                if use_soft:
                    loss = (
                        F.cross_entropy(cls_out, soft_targets)
                        + F.cross_entropy(dist_out, soft_targets)
                    ) / 2
                else:
                    loss = (
                        F.cross_entropy(cls_out, targets)
                        + F.cross_entropy(dist_out, targets)
                    ) / 2
                logits = cls_out
            else:
                logits = model(images)
                if use_soft:
                    loss = F.cross_entropy(logits, soft_targets)
                else:
                    loss = F.cross_entropy(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        # Accuracy computed against hard labels
        pred = logits.argmax(dim=1)
        correct += pred.eq(targets if not use_soft else soft_targets.argmax(dim=1)).sum().item()
        total += images.size(0)

        if (step + 1) % log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    use_dist = cfg["model"].get("use_dist_token", False)

    for images, targets in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=cfg["system"]["mixed_precision"]):
            if use_dist:
                output = model(images)  # averaged in eval mode
            else:
                output = model(images)
            loss = F.cross_entropy(output, targets)

        total_loss += loss.item() * images.size(0)
        correct += output.argmax(dim=1).eq(targets).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


# ════════════════════════════════════════════════════════════
#  Checkpoint
# ════════════════════════════════════════════════════════════

def save_checkpoint(state, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_acc = ckpt.get("best_acc", 0.0)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    print(f"  Resumed from {path} (epoch {start_epoch}, best_acc={best_acc:.2f}%)")
    return start_epoch, best_acc


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train DeiT-LinearTaylorIntegral on ImageNet-1K Parquet")
    parser.add_argument("--config", type=str, default="configs/deit_linear_taylor_integral.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None, choices=["tiny", "small", "base"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lazy_loading", action="store_true", help="Lazy parquet loading (less RAM)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)

    # ── Seed ──
    seed = cfg["system"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # ── Data ──
    img_size = cfg["model"]["img_size"]
    aug_cfg = cfg["train"]["augmentation"]
    data_format = cfg.get("data", {}).get("format", "parquet")

    if data_format == "huggingface":
        # Load directly from HuggingFace Hub via `datasets` library
        dataset_name = cfg["data"]["dataset_name"]
        cache_dir = cfg["data"].get("cache_dir", None)
        token = cfg["data"].get("token", True)  # True = use saved token
        trust_remote = cfg["data"].get("trust_remote_code", False)

        print(f"Loading HuggingFace dataset: {dataset_name}")
        train_dataset = build_hf_dataset(
            dataset_name, split="train", img_size=img_size,
            color_jitter=aug_cfg["color_jitter"],
            reprob=aug_cfg["reprob"],
            cache_dir=cache_dir, token=token,
            trust_remote_code=trust_remote,
        )
        val_dataset = build_hf_dataset(
            dataset_name, split="val", img_size=img_size,
            crop_ratio=cfg["eval"]["crop_ratio"],
            cache_dir=cache_dir, token=token,
            trust_remote_code=trust_remote,
        )
    else:
        # Load from local parquet files
        data_dir = cfg["train"]["data_dir"]
        lazy = cfg.get("data", {}).get("lazy", False)
        print(f"Loading data from: {data_dir} (lazy={lazy})")

        train_dataset = build_imagenet_dataset(
            data_dir, split="train", img_size=img_size,
            color_jitter=aug_cfg["color_jitter"],
            reprob=aug_cfg["reprob"],
            lazy=lazy,
        )
        val_dataset = build_imagenet_dataset(
            data_dir, split="val", img_size=img_size,
            crop_ratio=cfg["eval"]["crop_ratio"],
            lazy=lazy,
        )

    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples:   {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["system"]["num_workers"],
        pin_memory=cfg["system"]["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["system"]["num_workers"],
        pin_memory=cfg["system"]["pin_memory"],
    )

    # ── Model ──
    model = build_model(cfg)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    model_name = cfg["model"].get("name", "deit_linear_taylor_integral")
    print(f"Model: {model_name} [{cfg['model']['variant']}]  ({num_params:.1f}M params)")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=tuple(cfg["train"]["betas"]),
    )

    # ── Mixed precision ──
    scaler = torch.amp.GradScaler("cuda") if (cfg["system"]["mixed_precision"] and device.type == "cuda") else None

    # ── Mixup / CutMix ──
    mixup_fn = None
    if aug_cfg["mixup_alpha"] > 0 or aug_cfg["cutmix_alpha"] > 0:
        mixup_fn = MixupCutmix(
            mixup_alpha=aug_cfg["mixup_alpha"],
            cutmix_alpha=aug_cfg["cutmix_alpha"],
            prob=aug_cfg["mixup_prob"],
            switch_prob=aug_cfg["mixup_switch_prob"],
            num_classes=cfg["model"]["num_classes"],
            label_smoothing=cfg["train"]["label_smoothing"],
        )

    # ── Resume ──
    start_epoch = 0
    best_acc = 0.0
    if cfg.get("resume"):
        start_epoch, best_acc = load_checkpoint(cfg["resume"], model, optimizer, scaler)

    # ── Output directory with run counter and config name ──
    base_output_dir = cfg["system"]["output_dir"]
    config_name = os.path.splitext(os.path.basename(args.config))[0]  # e.g., "adaptive_token_agent_pet"
    run_num = 1
    output_dir = os.path.join(base_output_dir, f"run_{run_num}_{config_name}")
    while os.path.exists(output_dir):
        run_num += 1
        output_dir = os.path.join(base_output_dir, f"run_{run_num}_{config_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    total_epochs = cfg["train"]["epochs"]
    save_interval = cfg["system"]["save_interval"]
    
    # Early stopping
    early_stopping_patience = cfg["train"].get("early_stopping_patience", 15)
    early_stopping_metric = cfg["train"].get("early_stopping_metric", "val_acc")
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f" Training: epochs {start_epoch+1}–{total_epochs}, batch_size={cfg['train']['batch_size']}")
    print(f" Output directory: {output_dir}")
    print(f" Early stopping: metric={early_stopping_metric}, patience={early_stopping_patience}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, total_epochs):
        lr = adjust_learning_rate(optimizer, epoch, cfg)

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, mixup_fn, device, cfg, epoch
        )
        train_time = time.time() - t0

        val_loss, val_acc = validate(model, val_loader, device, cfg)

        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        print(
            f"Epoch {epoch+1}/{total_epochs} | "
            f"lr={lr:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
            f"best={best_acc:.2f}% | "
            f"time={train_time:.0f}s"
        )

        # Save checkpoints
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "best_acc": best_acc,
            "config": cfg,
        }

        if (epoch + 1) % save_interval == 0:
            save_checkpoint(state, output_dir, f"checkpoint_epoch_{epoch+1}.pth")
        if is_best:
            # Save best with metrics in filename and metadata
            best_ckpt_name = f"best_epoch{epoch+1}_val_acc{val_acc:.2f}_loss{val_loss:.4f}.pth"
            state["best_metrics"] = {
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "train_loss": train_loss,
            }
            save_checkpoint(state, output_dir, best_ckpt_name)
            # Also save as best.pth for convenience
            save_checkpoint(state, output_dir, "best.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"\n⛔ Early stopping: No improvement for {early_stopping_patience} epochs")
            print(f"Best {early_stopping_metric}: {best_acc:.2f}%\n")
            break

    # Save final
    save_checkpoint(state, output_dir, "last.pth")
    print(f"\nTraining complete. Best val accuracy: {best_acc:.2f}%")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    main()
