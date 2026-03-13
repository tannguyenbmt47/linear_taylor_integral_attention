"""
ImageNet-1K Parquet Dataset & Preprocessing

Handles ImageNet-1K stored as HuggingFace-style Parquet files where each row
contains an image (as raw bytes in a `image` column) and a label (`label` column).

Typical HuggingFace ImageNet-1K parquet structure:
    data/
        train/
            train-00000-of-01024.parquet
            train-00001-of-01024.parquet
            ...
        val/
            val-00000-of-00016.parquet
            ...

Each parquet file has columns: {"image": {"bytes": b"...", "path": "..."}, "label": int}
"""

import io
import glob
import os

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image


# ════════════════════════════════════════════════════════════
#  Transforms  (DeiT training recipe)
# ════════════════════════════════════════════════════════════

def build_train_transform(img_size=224, color_jitter=0.4, reprob=0.25):
    """DeiT-style training augmentation pipeline."""
    primary = [
        transforms.RandomResizedCrop(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
    ]
    if color_jitter > 0:
        primary.append(
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
            )
        )
    primary += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    if reprob > 0:
        primary.append(transforms.RandomErasing(p=reprob))
    return transforms.Compose(primary)


def build_val_transform(img_size=224, crop_ratio=0.875):
    """Standard validation transform: resize → center crop → normalize."""
    resize_size = int(img_size / crop_ratio)
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════

class ImageNetParquetDataset(Dataset):
    """
    Reads a single parquet file (or a list of row groups) containing
    ImageNet images as bytes + integer labels.

    Handles two common column formats:
      1. {"image": {"bytes": b"..."}, "label": int}      — HuggingFace default
      2. {"image": <binary>, "label": int}                — plain binary column
    """

    def __init__(self, parquet_path, transform=None):
        self.transform = transform
        table = pq.read_table(parquet_path)
        self.labels = table.column("label").to_pylist()

        # Extract image bytes — handle both struct and plain binary columns
        img_col = table.column("image")
        sample = img_col[0].as_py()
        if isinstance(sample, dict):
            # HuggingFace format: {"bytes": b"...", "path": "..."}
            self.image_bytes = [row.as_py()["bytes"] for row in img_col]
        else:
            # Plain binary
            self.image_bytes = [row.as_py() for row in img_col]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.image_bytes[idx])).convert("RGB")
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class ImageNetParquetShardDataset(Dataset):
    """
    Memory-efficient version that reads from parquet on-the-fly per shard,
    rather than loading everything into RAM at once.
    Reads row groups lazily.
    """

    def __init__(self, parquet_path, transform=None):
        self.parquet_path = parquet_path
        self.transform = transform
        pf = pq.ParquetFile(parquet_path)
        self.num_rows = pf.metadata.num_rows

        # Build row-group offsets for random access
        self._rg_offsets = []
        self._rg_lengths = []
        offset = 0
        for i in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(i).num_rows
            self._rg_offsets.append(offset)
            self._rg_lengths.append(n)
            offset += n

    def __len__(self):
        return self.num_rows

    def _find_row_group(self, idx):
        for i, (off, length) in enumerate(zip(self._rg_offsets, self._rg_lengths)):
            if idx < off + length:
                return i, idx - off
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx):
        rg_idx, local_idx = self._find_row_group(idx)
        pf = pq.ParquetFile(self.parquet_path)
        table = pf.read_row_group(rg_idx, columns=["image", "label"])

        label = table.column("label")[local_idx].as_py()
        img_val = table.column("image")[local_idx].as_py()
        img_bytes = img_val["bytes"] if isinstance(img_val, dict) else img_val

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ════════════════════════════════════════════════════════════
#  Builder: glob parquet shards → single Dataset
# ════════════════════════════════════════════════════════════

def build_imagenet_dataset(data_dir, split="train", img_size=224,
                           color_jitter=0.4, reprob=0.25, crop_ratio=0.875,
                           lazy=False):
    """
    Build an ImageNet dataset from a directory of parquet files.

    Args:
        data_dir:    Root directory containing train/ and val/ subdirs with .parquet files.
        split:       "train" or "val"
        img_size:    Target image size (default 224).
        color_jitter: Color jitter strength for training.
        reprob:      Random erasing probability for training.
        crop_ratio:  Center crop ratio for validation.
        lazy:        If True, use lazy shard reading (less RAM, slower).

    Returns:
        torch.utils.data.Dataset
    """
    split_dir = os.path.join(data_dir, split)
    parquet_files = sorted(glob.glob(os.path.join(split_dir, "*.parquet")))
    if not parquet_files:
        # Also check if parquets are directly in data_dir (flat layout)
        parquet_files = sorted(glob.glob(os.path.join(data_dir, f"{split}*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {split_dir} or {data_dir}/{split}*.parquet"
        )

    if split == "train":
        transform = build_train_transform(img_size, color_jitter, reprob)
    else:
        transform = build_val_transform(img_size, crop_ratio)

    DatasetCls = ImageNetParquetShardDataset if lazy else ImageNetParquetDataset
    datasets = [DatasetCls(p, transform=transform) for p in parquet_files]

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


# ════════════════════════════════════════════════════════════
#  Mixup / CutMix  (DeiT training recipe)
# ════════════════════════════════════════════════════════════

class MixupCutmix:
    """Mixup and CutMix data augmentation for batch-level mixing."""

    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0,
                 switch_prob=0.5, num_classes=1000, label_smoothing=0.1):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def __call__(self, images, targets):
        """
        Args:
            images: (B, C, H, W) tensor
            targets: (B,) integer labels
        Returns:
            mixed_images, soft_labels
        """
        B = images.size(0)
        # Convert to one-hot with label smoothing
        soft_targets = torch.zeros(B, self.num_classes, device=images.device)
        soft_targets.scatter_(1, targets.unsqueeze(1), 1.0)
        soft_targets = (
            soft_targets * (1 - self.label_smoothing)
            + self.label_smoothing / self.num_classes
        )

        if torch.rand(1).item() > self.prob:
            return images, soft_targets

        # Decide mixup or cutmix
        use_cutmix = torch.rand(1).item() < self.switch_prob and self.cutmix_alpha > 0

        if use_cutmix:
            lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item()
            rand_index = torch.randperm(B, device=images.device)
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.shape, lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (images.shape[-2] * images.shape[-1])
        else:
            if self.mixup_alpha <= 0:
                return images, soft_targets
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
            rand_index = torch.randperm(B, device=images.device)
            images = lam * images + (1 - lam) * images[rand_index]

        soft_targets = lam * soft_targets + (1 - lam) * soft_targets[rand_index]
        return images, soft_targets

    @staticmethod
    def _rand_bbox(size, lam):
        H, W = size[2], size[3]
        cut_rat = (1.0 - lam) ** 0.5
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)
        cx = torch.randint(0, H, (1,)).item()
        cy = torch.randint(0, W, (1,)).item()
        bbx1 = max(cx - cut_h // 2, 0)
        bby1 = max(cy - cut_w // 2, 0)
        bbx2 = min(cx + cut_h // 2, H)
        bby2 = min(cy + cut_w // 2, W)
        return bbx1, bby1, bbx2, bby2
