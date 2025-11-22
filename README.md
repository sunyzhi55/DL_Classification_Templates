---
date: 2025-11-22T18:46:00  
tags:
  - python
  - deep learning
---



<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=240&text=📦Deep%20Learning%20Classification%20Project&fontSize=40&fontAlign=50&fontColor=28F2E6&color=0:9AD6FF,50:C1A6FF,100:CFF7E6&desc=A%20Clean%20and%20Flexible%20PyTorch%20Classification%20Pipeline&descAlign=50&descAlignY=78&descSize=18&descColor=C8EFF0"/>
</p>



<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=240
  &text=📦Deep%20Learning%20Classification%20Project
  &fontSize=42&fontAlign=50&fontColor=C7FFF0
  &color=0:7AD0FF,50:8A6BFF,100:8EF6C2
  &desc=A%20Clean%20and%20Flexible%20PyTorch%20Classification%20Pipeline
  &descAlign=50&descAlignY=78&descSize=20&descColor=D6C8F9"/>
</p>



<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=240&text=📦Deep%20Learning%20Classification%20Project&fontSize=42&fontAlign=50&fontColor=B8FCE6&color=0:2F80ED,50:9055FF,100:00C9FF&desc=A%20Clean%20and%20Flexible%20PyTorch%20Classification%20Pipeline&descAlign=50&descAlignY=78&descSize=20&descColor=B2EBF2" alt="Mystic Aurora Theme"/>
</p>



<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=240
  &text=📦Deep%20Learning%20Classification%20Project
  &fontSize=42&fontAlign=50&fontColor=7FFFD4
  &color=0:0B2447,50:5B2B8A,100:00A586
  &desc=A%20Clean%20and%20Flexible%20PyTorch%20Classification%20Pipeline
  &descAlign=50&descAlignY=78&descSize=18&descColor=B1FBE4"/>
</p>





# 🌟 Deep Learning Image Classification Templates (PyTorch)



> **简洁 · 可扩展 · 工业级** —— 一个为研究与部署而生的通用图像分类项目模板。

---

## 📌 项目简介

这是一个基于 **PyTorch** 构建的**通用深度学习图像分类框架**，专为快速实验、模型对比与生产部署设计。项目提供：

- ✅ 多种主流视觉模型（ResNet, EfficientNet, EfficientViT, MetaFormer 等）  
- ✅ K-Fold 交叉验证支持  
- ✅ 灵活的数据加载（List 文件 / 文件夹格式）  
- ✅ 完善的日志记录、指标监控与训练可视化  
- ✅ 开箱即用的训练、测试与推理脚本  

无论你是学术研究者、算法工程师，还是刚入门深度学习的新手，该项目都能为你提供清晰、模块化且易于维护的代码基础。

---

## 🗂️ 目录结构

```text
project/
├── configs/           
│   └── config.py          # 全局配置解析与默认参数定义
├── data/              
│   ├── dataset.py         # 数据集加载器（支持 List 和 Folder 格式 + 增强策略）
│   └── ...                # （可扩展：CSV、HDF5 等）
├── models/            
│   ├── get_model.py       # 模型工厂函数（统一入口）
│   ├── ResNet.py
│   ├── EfficientNet.py
│   ├── EfficientViT.py
│   ├── MetaFormer.py
│   ├── PoolFormer.py
│   └── ...                # 支持无缝添加新架构
├── engine/            
│   └── trainer.py         # 训练/验证核心逻辑（含早停、调度器等）
├── utils/             
│   ├── basic.py           # 学习率调度、设备设置等基础工具
│   ├── observer.py        # 日志记录、指标跟踪、TensorBoard 支持
│   └── loss_function.py   # 自定义损失函数（如 LabelSmoothing）
├── main.py                # 主训练入口
├── infer.py               # 单图/批量推理脚本
├── test.py                # 模型评估脚本（准确率、混淆矩阵等）
└── README.md              # 你正在阅读的文档 ❤️
```

---

## 🛠 环境依赖

确保你的环境满足以下要求：

- **Python ≥ 3.8**
- **PyTorch ≥ 1.10**
- **torchvision**
- **scikit-learn**（用于 K-Fold 划分）
- **Pillow**（图像处理）
- **NumPy**
- **tqdm, tensorboard**（可选，用于进度条与日志可视化）

推荐使用 `conda` 或 `venv` 创建独立环境：

```bash
pip install torch torchvision scikit-learn pillow numpy tqdm tensorboard
```

---

## 🚀 快速开始

### 1️⃣ 数据准备

项目默认支持 **List 文件格式**（每行：`图像路径 类别ID`）：

以`Oxford 102 Flowers`数据集为例：



```text
/path/to/flower_001.jpg 0
/path/to/flower_042.jpg 1
...
```

准备 `train.txt` 和 `test.txt`（或仅 `train.txt`，内部自动划分验证集）。

> 💡 提示：类别 ID 应为从 `0` 开始的连续整数。

---

### 2️⃣ 启动训练

所有参数均可通过命令行覆盖，默认值定义于 `configs/config.py`。

#### 🔹 基础训练
```bash
python main.py \
  --data_dir /your/image/root \
  --train_eval_label_file_path /path/to/train.txt \
  --num_classes 102 \
  --exp_name flowers_resnet50
```

#### 🔹 K-Fold 交叉验证（例如 5 折）
```bash
python main.py --k_fold 5 --epochs 100 --batch_size 32
```

#### 🔹 单次训练（80/20 自动划分）
```bash
python main.py --k_fold 0 --epochs 50
```

#### 🔸 常用参数说明
| 参数 | 说明 |
|------|------|
| `--exp_name` | 实验名称（用于日志和模型保存目录） |
| `--device` | 设备（如 `cuda:0`, `cpu`） |
| `--model_name` | 模型名称（需在 `get_model.py` 中注册） |
| `--lr` | 初始学习率（默认 `1e-3`） |
| `--output_dir` | 输出根目录（日志、权重、配置备份） |

---

### 3️⃣ 模型推理

对单张或多张图像进行预测：

```bash
python infer.py \
  --image img1.jpg img2.jpg \
  --checkpoint best_model.pth \
  --num_classes 102 \
  --device cuda:0
```

输出示例：
```
./img1.jpg → class 17 (probability: 0.92)
./img2.jpg → class 42 (probability: 0.88)
```

---

## 🧠 支持的模型架构

| 模型 | 文件 | 特点 |
|------|------|------|
| **ResNet** | `ResNet.py` | 经典残差网络，稳定可靠 |
| **EfficientNet** | `EfficientNet.py` | 高效缩放，精度/速度平衡 |
| **EfficientViT** | `EfficientViT.py` | 轻量级 Vision Transformer |
| **MetaFormer** | `MetaFormer.py` | 统一 CNN/Transformer 的骨干 |
| **PoolFormer** | `PoolFormer.py` | 基于池化的纯 Transformer 替代方案 |

> 所有模型均支持 ImageNet 预训练权重加载（若可用）。

---

## 🛠️ 扩展指南

### ➕ 添加新模型
1. 在 `models/` 下创建 `your_model.py`，定义 `YourModel(...)` 类。
2. 在 `models/get_model.py` 中导入并注册：
   ```python
   elif model_name == "your_model":
       return YourModel(num_classes=num_classes, ...)
   ```
3. 启动时指定 `--model_name your_model` 即可。

### ➕ 自定义数据集
1. 在 `data/dataset.py` 中继承 `torch.utils.data.Dataset`。
2. 实现 `__len__` 和 `__getitem__` 方法。
3. 在 `main.py` 中根据参数选择数据集类。

### ➕ 修改训练流程
- 所有训练逻辑封装在 `engine/trainer.py`。
- 可自定义：
  - 损失函数（修改 `loss_fn`）
  - 评估指标（如 Top-1/Top-5 Acc）
  - 日志频率、早停策略、学习率调度器等

---

## 📬 贡献与反馈

欢迎提交 Issue 或 Pull Request！如果你觉得这个项目对你有帮助，请 ⭐ Star 支持！

---

> **Made with ❤️ and PyTorch** 
> © 2025 Deep Learning Classification Project — MIT License

---

✅ **现在就克隆项目，开启你的图像分类之旅吧！**

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=9580FF&text=✨%20Enjoy%20Building%20Your%20Model!%20✨&fontColor=FFFFFF&fontSize=25&height=80"/>
</p>