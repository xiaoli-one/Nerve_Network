#!/usr/bin/env python3
# WM811K CNN 晶圆缺陷分类
# 快速模式：通道192，两个残差块：/Users/agiuser/py3/bin/python wm811k_cnn_assignment.py
# strong模式：通道384，7个残差块，添加SEBlock注意力：/Users/agiuser/py3/bin/python wm811k_cnn_assignment.py --model strong
# 如果要使用无标注数据 加：--use-unlabeled
# 快速模式：无标注
# 现在只剩减残差加准确的
# 5个epoch（减残差加准确）：  Accuracy: 0.9547， Macro F1: 0.6506，Weighted F1: 0.9547， loss: 0.1814
# 10个epoch： Accuracy: 0.9496， Macro F1   : 0.7138，Weighted F1: 0.9539, loss:5epoch 0.0416， 10epoch 0.0240
# 10个epoch（减残差加准确）：没做

# 为了提高精度，减少了残差，关了均衡采样：WeightedSampler + 类别权重 + FocalLoss，默认是sampler_power, weight_power, focal_gamma = 0.0, 0.0, 0.0
# 需要照顾缺陷类就回调：sampler_power, weight_power, focal_gamma = 0.2, 0.05, 0.3 
# 如果像要更高精度，可以把调--no-bias：--none-bias 0.5
# 如果发现F1掉太多，可以回调点：--none-bias 0.2

import argparse
import copy
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def label_to_str(x):
    if isinstance(x, str):
        return x
    if isinstance(x, np.ndarray):
        return "" if x.size == 0 else label_to_str(x.reshape(-1)[0])
    if isinstance(x, (list, tuple)):
        return "" if len(x) == 0 else label_to_str(x[0])
    return "" if pd.isna(x) else str(x)


def load_data(path, keep_unlabeled=False):
    print(f"读取数据：{path}")
    df = pd.read_pickle(path)
    labels = np.array([label_to_str(x) for x in df["failureType"].to_numpy()])
    split = np.array([label_to_str(x) for x in df["trainTestLabel"].to_numpy()])
    mask = np.isin(split, ["Training", "Test"])

    maps = df.loc[mask, "waferMap"].to_numpy()
    unlabeled_maps = df.loc[~mask, "waferMap"].to_numpy() if keep_unlabeled else None
    labels = labels[mask]
    split = split[mask]
    name_to_id = {name: i for i, name in enumerate(CLASSES)}
    y = np.array([name_to_id[name] for name in labels], dtype=np.int64)
    del df

    print(f"有标注样本：{len(y)}")
    for name in CLASSES:
        print(f"{name:10s}: {np.sum(labels == name):7d}")
    print(f"训练集：{np.sum(split == 'Training')}，测试集：{np.sum(split == 'Test')}")
    if keep_unlabeled:
        print(f"无标注样本：{len(unlabeled_maps)}")
    return maps, y, split, unlabeled_maps


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    # 本机 MPS 曾出现训练数值不稳定；自动模式优先 CUDA，否则 CPU。
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def limit_samples(idx, y, n, seed):
    if n is None or n <= 0 or n >= len(idx):
        return idx
    cls, cnt = np.unique(y[idx], return_counts=True)
    if n < len(cls) or cnt.min() < 2:
        rng = np.random.default_rng(seed)
        return rng.choice(idx, size=n, replace=False)
    return train_test_split(idx, train_size=n, stratify=y[idx], random_state=seed)[0]


def make_split(y, split, val_ratio, seed, max_train=None, max_val=None, max_test=None):
    train_all = np.where(split == "Training")[0]
    test_idx = np.where(split == "Test")[0]
    train_idx, val_idx = train_test_split(
        train_all,
        test_size=val_ratio,
        stratify=y[train_all],
        random_state=seed,
    )
    train_idx = limit_samples(train_idx, y, max_train, seed)
    val_idx = limit_samples(val_idx, y, max_val, seed + 1)
    test_idx = limit_samples(test_idx, y, max_test, seed + 2)
    return train_idx, val_idx, test_idx


def wafer_to_tensor(arr, size=96, augment=False):
    arr = np.asarray(arr, dtype=np.uint8)
    if augment:
        arr = np.rot90(arr, random.randint(0, 3))
        if random.random() < 0.5:
            arr = np.flip(arr, 0)
        if random.random() < 0.5:
            arr = np.flip(arr, 1)
        arr = np.ascontiguousarray(arr)

    h, w = arr.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    square[top : top + h, left : left + w] = arr

    # 0/1/2 三种 die 状态做 one-hot，比直接灰度输入更容易学。
    x = np.stack([square == 0, square == 1, square == 2]).astype(np.float32)
    x = torch.from_numpy(x)
    return F.interpolate(x[None], (size, size), mode="nearest")[0]


class WaferDataset(Dataset):
    def __init__(self, maps, y, idx, size=96, augment=False):
        self.maps = maps
        self.y = y
        self.idx = np.asarray(idx)
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        return wafer_to_tensor(self.maps[j], self.size, self.augment), torch.tensor(self.y[j], dtype=torch.long)


class UnlabeledDataset(Dataset):
    def __init__(self, maps, idx, size=96):
        self.maps = maps
        self.idx = np.asarray(idx)
        self.size = size

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        return wafer_to_tensor(self.maps[j], self.size, augment=False), torch.tensor(j, dtype=torch.long)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        hidden = max(16, ch // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch)
        self.drop = nn.Dropout2d(drop) if drop else nn.Identity()
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.se(y)
        y = self.drop(y)
        return F.silu(y + self.skip(x), inplace=True)


class FastWaferNet(nn.Module):
    def __init__(self, num_classes=9, dropout=0.20):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(3, 32),
            ConvBNAct(32, 32),
            nn.MaxPool2d(2),
            ConvBNAct(32, 64),
            ConvBNAct(64, 64),
            nn.MaxPool2d(2),
            ResBlock(64, 128, stride=2, drop=0.03),
            ResBlock(128, 128, drop=0.03),
            ConvBNAct(128, 192, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class WaferNet(nn.Module):
    def __init__(self, num_classes=9, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(3, 32),
            ConvBNAct(32, 64),
            ResBlock(64, 64, drop=0.02),
            ResBlock(64, 128, stride=2, drop=0.03),
            ResBlock(128, 128, drop=0.03),
            ResBlock(128, 256, stride=2, drop=0.05),
            ResBlock(256, 256, drop=0.05),
            ResBlock(256, 384, stride=2, drop=0.08),
            ResBlock(384, 384, drop=0.08),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.0, smoothing=0.02):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, weight=self.weight, reduction="none", label_smoothing=self.smoothing)
        pt = torch.softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
        return (((1 - pt).clamp_min(1e-6) ** self.gamma) * ce).mean()


def class_weights(y, train_idx, power):
    counts = np.bincount(y[train_idx], minlength=len(CLASSES)).astype(np.float32)
    weights = (counts.sum() / (len(CLASSES) * np.maximum(counts, 1))) ** power
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def sampler_for(y, train_idx, power):
    if power <= 0:
        return None
    counts = np.bincount(y[train_idx], minlength=len(CLASSES)).astype(np.float32)
    weights = (1 / np.maximum(counts, 1) ** power)[y[train_idx]]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(train_idx), replacement=True)


def tta_forward(model, x, tta=1, none_bias=0.0):
    if tta <= 1:
        out = model(x)
        out[:, CLASSES.index("none")] += none_bias
        return out
    logits = []
    for k in range(4):
        xx = torch.rot90(x, k, dims=(-2, -1)) if k else x
        logits.append(model(xx))
    if tta >= 8:
        flip = torch.flip(x, dims=(-1,))
        for k in range(4):
            xx = torch.rot90(flip, k, dims=(-2, -1)) if k else flip
            logits.append(model(xx))
    out = torch.stack(logits).mean(0)
    out[:, CLASSES.index("none")] += none_bias
    return out


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device, epoch, epochs):
    model.train()
    total_loss = total_correct = total = skipped = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", ncols=118)
    use_amp = device.type == "cuda"

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x)
            loss = loss_fn(out, y)
        if not torch.isfinite(loss):
            skipped += 1
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=False)
        if not torch.isfinite(grad_norm):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = y.size(0)
        total += bs
        total_loss += loss.item() * bs
        total_correct += (out.argmax(1) == y).sum().item()
        pbar.set_postfix(loss=f"{total_loss / max(total, 1):.4f}", acc=f"{total_correct / max(total, 1):.4f}")

    if skipped:
        print(f"警告：跳过 {skipped} 个非有限 loss/梯度 batch")
    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, desc="Evaluate", tta=1, none_bias=0.0):
    model.eval()
    total_loss, y_true, y_pred = 0, [], []
    for x, y_cpu in tqdm(loader, desc=desc, ncols=118, leave=False):
        y_true.extend(y_cpu.numpy())
        x, y = x.to(device), y_cpu.to(device)
        out = tta_forward(model, x, tta, none_bias)
        total_loss += loss_fn(out, y).item() * y.size(0)
        y_pred.extend(out.argmax(1).cpu().numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        "loss": total_loss / len(y_true),
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=np.arange(len(CLASSES)), average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=np.arange(len(CLASSES)), average="weighted", zero_division=0),
        "true": y_true,
        "pred": y_pred,
    }


@torch.no_grad()
def make_pseudo_labels(model, unlabeled_maps, args, device):
    model.eval()
    if unlabeled_maps is None or len(unlabeled_maps) == 0:
        return None, None

    all_idx = np.arange(len(unlabeled_maps))
    if args.pseudo_pool and args.pseudo_pool < len(all_idx):
        rng = np.random.default_rng(args.seed)
        all_idx = rng.choice(all_idx, size=args.pseudo_pool, replace=False)

    loader = DataLoader(
        UnlabeledDataset(unlabeled_maps, all_idx, args.image_size),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    chosen_idx, chosen_y, chosen_conf = [], [], []
    for x, raw_idx in tqdm(loader, desc="Pseudo label", ncols=118):
        x = x.to(device)
        logits = tta_forward(model, x, tta=1, none_bias=args.none_bias)
        prob = torch.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)
        keep = conf >= args.pseudo_threshold
        if keep.any():
            chosen_idx.append(raw_idx[keep.cpu()].numpy())
            chosen_y.append(pred[keep].cpu().numpy())
            chosen_conf.append(conf[keep].cpu().numpy())

    if not chosen_idx:
        print("没有达到阈值的无标注样本，本轮不加入伪标签。")
        return None, None

    chosen_idx = np.concatenate(chosen_idx)
    chosen_y = np.concatenate(chosen_y).astype(np.int64)
    chosen_conf = np.concatenate(chosen_conf)

    if args.max_pseudo and args.max_pseudo < len(chosen_idx):
        order = np.argsort(-chosen_conf)[: args.max_pseudo]
        chosen_idx, chosen_y, chosen_conf = chosen_idx[order], chosen_y[order], chosen_conf[order]

    print(f"加入伪标签样本：{len(chosen_idx)}，平均置信度：{chosen_conf.mean():.4f}")
    counts = np.bincount(chosen_y, minlength=len(CLASSES))
    for name, count in zip(CLASSES, counts):
        if count:
            print(f"  pseudo {name:10s}: {count}")
    return chosen_idx, chosen_y


def rebuild_train_loader(maps, y, train_idx, args, sampler_power, device):
    train_ds = WaferDataset(maps, y, train_idx, args.image_size, augment=True)
    return DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler_for(y, train_idx, sampler_power),
        shuffle=sampler_power <= 0,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def score_to_select(metric, mode):
    if mode == "balanced":
        return 0.5 * metric["acc"] + 0.5 * metric["macro_f1"]
    return metric["acc"]


def plot_class_distribution(y, train_idx, test_idx):
    train_counts = np.bincount(y[train_idx], minlength=len(CLASSES))
    test_counts = np.bincount(y[test_idx], minlength=len(CLASSES))
    x = np.arange(len(CLASSES))
    plt.figure(figsize=(11, 4))
    plt.bar(x - 0.2, train_counts, width=0.4, label="train")
    plt.bar(x + 0.2, test_counts, width=0.4, label="test")
    plt.yscale("log")
    plt.xticks(x, CLASSES, rotation=35, ha="right")
    plt.ylabel("count (log)")
    plt.title("Class Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_curves(history):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_loss"], "o-", label="train loss")
    plt.plot(epochs, history["val_loss"], "o-", label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_acc"], "o-", label="train acc")
    plt.plot(epochs, history["val_acc"], "o-", label="val acc")
    plt.plot(epochs, history["val_f1"], "o-", label="val macro F1")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.ylim(0, 1.02)
    plt.title("Accuracy / F1")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_result(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))
    cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="recall normalized")
    plt.xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right")
    plt.yticks(range(len(CLASSES)), CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    f1 = f1_score(y_true, y_pred, labels=np.arange(len(CLASSES)), average=None, zero_division=0)
    plt.figure(figsize=(10, 4))
    plt.bar(CLASSES, f1)
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1.02)
    plt.ylabel("F1")
    plt.title("Per-class F1")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/WM811K.pkl"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--model", default="fast", choices=["fast", "strong"])
    parser.add_argument("--mode", default="acc", choices=["acc", "balanced"], help="acc 优先总体精度；balanced 兼顾 macro F1")
    parser.add_argument("--tta", type=int, default=4, choices=[1, 4, 8])
    parser.add_argument("--none-bias", type=float, default=0.35, help="推理时给 none 类加的 logit 偏置，提高总体 accuracy")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-unlabeled", action="store_true", help="使用无标注数据做高置信度伪标签训练")
    parser.add_argument("--pseudo-start-epoch", type=int, default=3, help="第几个 epoch 结束后生成伪标签")
    parser.add_argument("--pseudo-threshold", type=float, default=0.995, help="伪标签最大概率阈值")
    parser.add_argument("--max-pseudo", type=int, default=30000, help="最多加入多少个伪标签样本，0 表示不限制")
    parser.add_argument("--pseudo-pool", type=int, default=100000, help="最多扫描多少个无标注样本，0 表示扫描全部")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    seed_all(args.seed)
    device = choose_device(args.device)
    maps, y, split, unlabeled_maps = load_data(args.data, keep_unlabeled=args.use_unlabeled)
    train_idx, val_idx, test_idx = make_split(
        y, split, args.val_ratio, args.seed, args.max_train, args.max_val, args.max_test
    )

    if args.mode == "acc":
        sampler_power, weight_power, focal_gamma = 0.0, 0.0, 0.0
    else:
        sampler_power, weight_power, focal_gamma = 0.75, 0.25, 1.2

    print(f"实际使用：train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"设备：{device}")
    print(f"模型：{args.model}（fast 单 epoch 快；strong 精度上限更高但慢）")
    print("基础提升方法：增强旋转/翻转、one-hot 输入、残差 CNN、SE 注意力、OneCycleLR、早停、TTA")
    if args.mode == "acc":
        print("accuracy 优先：默认关闭类别权重、WeightedSampler、FocalLoss，减少把 none 误判为缺陷类")
    else:
        print("balanced 优先：启用类别权重、WeightedSampler、FocalLoss，提高少数缺陷类识别")
    print(f"类别均衡参数：sampler_power={sampler_power}, weight_power={weight_power}, focal_gamma={focal_gamma}")
    print(f"优化目标：{args.mode}，TTA={args.tta}，none_bias={args.none_bias}")
    print(f"测试集全预测 none 的 accuracy baseline：{np.mean(y[test_idx] == CLASSES.index('none')):.4f}")

    if not args.no_plots:
        plot_class_distribution(y, train_idx, test_idx)

    val_ds = WaferDataset(maps, y, val_idx, args.image_size, augment=False)
    test_ds = WaferDataset(maps, y, test_idx, args.image_size, augment=False)

    train_loader = rebuild_train_loader(maps, y, train_idx, args, sampler_power, device)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)

    model = (FastWaferNet(len(CLASSES)) if args.model == "fast" else WaferNet(len(CLASSES))).to(device)
    weights = class_weights(y, train_idx, weight_power).to(device)
    loss_fn = FocalLoss(weights, gamma=focal_gamma, smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.12,
        div_factor=20,
        final_div_factor=200,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score, best_epoch, no_improve = -1, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    start = time.time()
    pseudo_added = False

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, scaler, device, epoch, args.epochs
        )
        val = evaluate(model, val_loader, loss_fn, device, desc="Validate", tta=1, none_bias=args.none_bias)
        score = score_to_select(val, args.mode)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        history["val_f1"].append(val["macro_f1"])

        if score > best_score:
            best_score, best_epoch, no_improve = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model": best_state, "classes": CLASSES, "args": vars(args)}, "wm811k_best.pt")
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val['loss']:.4f}, val_acc={val['acc']:.4f}, "
            f"val_macro_f1={val['macro_f1']:.4f}, best_score={best_score:.4f}@{best_epoch}"
        )

        if args.patience > 0 and no_improve >= args.patience:
            print(f"早停：连续 {args.patience} 个 epoch 没有提升")
            break

        if args.use_unlabeled and (not pseudo_added) and epoch >= args.pseudo_start_epoch:
            model.load_state_dict(best_state)
            pseudo_idx, pseudo_y = make_pseudo_labels(model, unlabeled_maps, args, device)
            if pseudo_idx is not None:
                old_n = len(maps)
                maps = np.concatenate([maps, unlabeled_maps[pseudo_idx]])
                y = np.concatenate([y, pseudo_y])
                train_idx = np.concatenate([train_idx, np.arange(old_n, old_n + len(pseudo_idx))])
                train_loader = rebuild_train_loader(maps, y, train_idx, args, sampler_power, device)
                weights = class_weights(y, train_idx, weight_power).to(device)
                loss_fn = FocalLoss(weights, gamma=focal_gamma, smoothing=0.02)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=args.lr,
                    epochs=max(1, args.epochs - epoch),
                    steps_per_epoch=max(1, len(train_loader)),
                    pct_start=0.12,
                    div_factor=20,
                    final_div_factor=200,
                )
                pseudo_added = True
                print(f"伪标签已加入训练集，现在 train={len(train_idx)}。")

    print(f"训练耗时：{(time.time() - start) / 60:.1f} 分钟")
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, loss_fn, device, desc=f"Test TTA={args.tta}", tta=args.tta, none_bias=args.none_bias)

    print("\n测试集结果")
    print(f"Accuracy   : {test['acc']:.4f}")
    print(f"Macro F1   : {test['macro_f1']:.4f}")
    print(f"Weighted F1: {test['weighted_f1']:.4f}")
    print(classification_report(test["true"], test["pred"], labels=np.arange(len(CLASSES)), target_names=CLASSES, zero_division=0))

    if not args.no_plots:
        plot_curves(history)
        plot_result(test["true"], test["pred"])


if __name__ == "__main__":
    main()
