import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


def load_class_map(path: str):
    class_to_idx = json.loads(Path(path).read_text(encoding="utf-8"))
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def build_model(num_classes: int):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


def preprocess_image(image_path: str):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    class_to_idx, idx_to_class = load_class_map(args.class_map)

    model = build_model(num_classes=len(class_to_idx))
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image_tensor = preprocess_image(args.image).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)

    top_probs, top_indices = probs.topk(args.topk, dim=1)

    print("预测结果:")
    for p, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
        label = idx_to_class[idx]
        print(f"- 类别: {label}, 概率: {p:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch 图像分类推理脚本")
    parser.add_argument("--image", type=str, required=True, help="待预测图片路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径（best_model.pt）")
    parser.add_argument("--class_map", type=str, required=True, help="类别映射 JSON 路径")
    parser.add_argument("--topk", type=int, default=3, help="输出前 K 个结果")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
