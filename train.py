import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


def build_local_dataloaders(data_dir: str, batch_size: int, num_workers: int = 2):
    train_transform, val_transform = build_transforms()

    train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_set, val_set, train_loader, val_loader


def build_cloud_dataloaders(cloud_dataset: str, cloud_data_dir: str, batch_size: int, num_workers: int = 2):
    train_transform, val_transform = build_transforms()

    if cloud_dataset.lower() != "cifar10":
        raise ValueError("当前云端样例仅支持 --cloud_dataset cifar10。")

    train_set = datasets.CIFAR10(root=cloud_data_dir, train=True, download=True, transform=train_transform)
    val_set = datasets.CIFAR10(root=cloud_data_dir, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_set, val_set, train_loader, val_loader


def build_model(num_classes: int, pretrained: bool = True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def save_loss_curve(train_losses, val_losses, output_path: Path):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_source == "local":
        train_set, _, train_loader, val_loader = build_local_dataloaders(
            args.data_dir, args.batch_size, args.num_workers
        )
    else:
        train_set, _, train_loader, val_loader = build_cloud_dataloaders(
            args.cloud_dataset,
            args.cloud_data_dir,
            args.batch_size,
            args.num_workers,
        )

    class_names = train_set.classes
    if len(class_names) < 2:
        raise ValueError("至少需要 2 个类别用于分类任务。")

    model = build_model(num_classes=len(class_names), pretrained=not args.no_pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_model_path = output_dir / "best_model.pt"

    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ 保存最佳模型到: {best_model_path}")

    loss_curve_path = output_dir / "loss_curve.png"
    save_loss_curve(train_losses, val_losses, loss_curve_path)

    class_to_idx_path = output_dir / "class_to_idx.json"
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    class_to_idx_path.write_text(json.dumps(class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")
    print(f"类别映射已保存到: {class_to_idx_path}")
    print(f"Loss 曲线已保存到: {loss_curve_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch 图像分类训练脚本")
    parser.add_argument("--dataset_source", type=str, choices=["local", "cloud"], default="local", help="数据来源")
    parser.add_argument("--data_dir", type=str, default="data", help="本地数据根目录，包含 train/ 和 val/")
    parser.add_argument("--cloud_dataset", type=str, default="cifar10", help="云端样例数据集（当前支持 cifar10）")
    parser.add_argument("--cloud_data_dir", type=str, default="./cloud_data", help="云端数据下载缓存目录")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker 数")
    parser.add_argument("--no_pretrained", action="store_true", help="不使用预训练权重")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
