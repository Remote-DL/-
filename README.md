# Python 深度学习图像分类示例（PyTorch）

这是一个可直接运行的图像分类模板，包含：

- `train.py`：训练与验证（支持本地/云端样例数据）
- `predict.py`：单张图片推理
- `requirements.txt`：依赖

## 1. 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 本地数据集目录结构（ImageFolder）

请按 `ImageFolder` 结构组织数据：

```text
data/
  train/
    cat/
      001.jpg
      002.jpg
    dog/
      101.jpg
  val/
    cat/
      201.jpg
    dog/
      301.jpg
```

- `train` 和 `val` 下的子目录名就是类别名。
- 至少需要 2 个类别。

## 3. 本地数据训练

```bash
python train.py \
  --dataset_source local \
  --data_dir ./data \
  --epochs 10 \
  --batch_size 32 \
  --lr 1e-3 \
  --output_dir ./outputs
```

## 4. 云端样例训练（自动下载 CIFAR10）

> 适合快速查看完整训练流程，不需要你先准备本地图片数据。

```bash
python train.py \
  --dataset_source cloud \
  --cloud_dataset cifar10 \
  --cloud_data_dir ./cloud_data \
  --epochs 5 \
  --batch_size 64 \
  --output_dir ./outputs_cifar10
```

## 5. 训练产物

- `outputs/best_model.pt`：最佳权重
- `outputs/class_to_idx.json`：类别映射
- `outputs/loss_curve.png`：训练过程 loss 曲线（Train/Val）

## 6. 进行预测

```bash
python predict.py \
  --image ./demo.jpg \
  --model_path ./outputs/best_model.pt \
  --class_map ./outputs/class_to_idx.json \
  --topk 3
```

## 7. 常见问题

1. **显存不足**：减小 `--batch_size`，或禁用预训练。  
2. **类别不平衡**：可在 `train.py` 中增加加权采样器或损失权重。  
3. **精度不高**：增加数据增强、训练轮数，或更换 backbone（如 ResNet50 / EfficientNet）。
