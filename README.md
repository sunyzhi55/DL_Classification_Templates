---
date: 2025-11-22T18:46:00  
tags:
  - python
  - deep learning
---



<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=240&text=📦Deep%20Learning%20Classification%20Project&fontSize=40&fontAlign=50&fontColor=28F2E6&color=0:9AD6FF,50:C1A6FF,100:CFF7E6&desc=A%20Clean%20and%20Flexible%20PyTorch%20Classification%20Pipeline&descAlign=50&descAlignY=78&descSize=18&descColor=C8EFF0&animation=twinkling"/>
</p>





<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=240&text=📦Deep%20Learning%20Classification%20Project&fontSize=42&fontAlign=50&fontColor=C7FFF0&color=0:7AD0FF,50:8A6BFF,100:8EF6C2&desc=A%20Clean%20and%20Flexible%20PyTorch%20Classification%20Pipeline&descAlign=50&descAlignY=78&descSize=20&descColor=D6C8F9&animation=twinkling"/>
</p>



<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=240&text=📦Deep%20Learning%20Classification%20Project&fontSize=42&fontAlign=50&fontColor=7FFFD4&color=0:0B2447,50:5B2B8A,100:00A586&desc=A%20Clean%20and%20Flexible%20PyTorch%20Classification%20Pipeline&descAlign=50&descAlignY=78&descSize=18&descColor=B1FBE4&animation=twinkling"/>
</p>




# 🌟 Deep Learning Image Classification Templates (PyTorch)



> **简洁 · 可扩展 · 工业级** —— 一个为研究与部署而生的通用图像分类项目模板。

---

## 📌 1. 项目简介

这是一个基于 **PyTorch** 构建的**通用深度学习图像分类框架**，专为快速实验、模型对比与生产部署设计。项目提供：

- ✅ 多种主流视觉模型（ResNet, EfficientNet, EfficientViT, MetaFormer 等）  
- ✅ K-Fold 交叉验证支持  
- ✅ 灵活的数据加载（List 文件 / 文件夹格式）  
- ✅ 完善的日志记录、指标监控与训练可视化（TensorBoard + SwanLab）  
- ✅ 开箱即用的训练、测试与推理脚本
- ✅ **完整的可复现性保证**（全局随机种子设置）  

无论你是学术研究者、算法工程师，还是刚入门深度学习的新手，该项目都能为你提供清晰、模块化且易于维护的代码基础。

---

## 🗂️ 2. 目录结构

```text
project/
├── configs/           
│   ├── config.py              # 全局配置解析与默认参数定义
│   └── experiments_object.py  # 实验配置字典
├── data/              
│   └── dataset.py             # 数据集加载器（支持 List 和 Folder 格式 + 增强策略）
├── models/            
│   ├── get_model.py           # 模型工厂函数（统一入口）
│   ├── ResNet.py
│   ├── EfficientNet.py
│   └── ...                    # 支持无缝添加新架构
├── engine/            
│   └── trainer.py             # 训练/验证核心逻辑（含早停、调度器等）
├── experiment_results/        # 记录每次实验的日志
├── utils/
│   ├── basic.py               # 学习率调度、设备设置等基础工具
│   ├── loss_function.py       # 自定义损失函数
│   ├── model_stats.py         # 模型参数与 FLOPs 计算工具
│   ├── observer.py            # 日志记录、指标跟踪、实验可视化
│   ├── swanlab_logger.py      # SwanLab 实验跟踪模块（可选）
│   └── reproducibility.py     # 可复现性工具（随机种子设置）
├── main.py                    # 主训练入口
├── README.md                  # 你正在阅读的文档 ❤️
├── infer.py                   # 单图/批量推理脚本
├── test.py                    # 模型评估脚本
└── requirements.txt           # 依赖管理

```

---

## 🛠 3. 环境依赖

确保你的环境满足以下要求：

- **Python ≥ 3.8**
- **PyTorch ≥ 1.10**
- **torchvision**
- **scikit-learn**（用于 K-Fold 划分）
- **Pillow**（图像处理）
- **NumPy**
- **tqdm**（用于进度条显示）
- **tensorboard**（实验可视化）
- **matplotlib, seaborn**（用于保存混淆矩阵可视化）
- **ptflops**（用于计算模型 FLOPs / MACs）
- **swanlab**（可选，用于云端实验跟踪）

推荐使用 `conda` 或 `venv` 创建独立环境：

```bash
# 安装核心依赖
pip install torch torchvision scikit-learn pillow numpy tqdm tensorboard matplotlib seaborn ptflops

# （可选）安装 SwanLab 用于实验跟踪
pip install swanlab

# 或使用 requirements.txt 一键安装所有依赖
pip install -r requirements.txt

```
---

## 🚀 4. 快速开始

### 1️⃣ 4.1 数据准备

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

### 2️⃣ 4.2 启动训练

项目采用**字典配置驱动**的方式管理实验。所有的实验配置都集中在 `configs/experiments_object.py` 文件中。

#### 🔹 步骤 1：定义实验配置
打开 `configs/experiments_object.py`，在 `experiments` 字典中添加你的实验配置。你可以复制现有的配置并修改参数：

```python
experiments = {
    "My_New_Experiment": {
        # Model
        "model": resnet50,  # 直接引用模型类
        "num_classes": 10,
        
        # Dataset
        "dataset": MyCustomDataset, # 直接引用数据集类
        "data_dir": "/path/to/data",
        
        # Training
        "batch_size": 32,
        "lr": 1e-3,
        "epochs": 100,
        # ... 其他参数
    }
}
```

#### 🔹 步骤 2：运行实验
使用 `--exp_name` 参数指定你要运行的实验名称：

```bash
python main.py --exp_name My_New_Experiment
```

程序会自动加载字典中定义的所有参数（模型、数据集、优化器、超参数等），并覆盖默认配置。

**✨ 可复现性保证：**
- 程序会自动使用配置中的 `seed` 参数设置全局随机种子
- 输出会显示：`✅ Global random seed set to: XXX`
- 相同配置的多次运行将产生完全相同的结果

```python
# 配置示例（包含 seed）
"My_Experiment": {
    "seed": 42,  # 随机种子，确保可复现性
    "model_name": "resnet34",
    "batch_size": 64,
    # ... 其他参数
}
```

#### 🔹 为什么使用字典配置？
- **集中管理**：所有实验的超参数一目了然，方便对比和复现。
- **灵活性**：可以直接在配置中引用 Python 对象（如模型类、数据集类、优化器类），而不仅仅是字符串。
- **版本控制**：配置文件本身就是代码的一部分，方便使用 Git 进行版本控制。

---

### 3️⃣ 4.3 模型推理

1、对单张或多张图像进行预测：

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
2、对文件夹中的所有图像进行批量预测：

```bash
python infer.py \
  --folder /path/to/image/folder \
  --checkpoint best_model.pth \
  --num_classes 102 \
  --device cuda:0
```
输出示例：
```
/your/image/folder/img1.jpg → class 17 (probability: 0.92)
/your/image/folder/img2.jpg → class 42 (probability: 0.88)
```


---

## 🔄 5. 可复现性与实验管理

### 🎯 完整的可复现性保证

项目已实现工业级的可复现性设置，确保实验结果可以被准确复现：

```python
# 所有随机源都被控制
✅ Python random
✅ NumPy random  
✅ PyTorch CPU random
✅ PyTorch GPU random (CUDA)
✅ DataLoader shuffle
✅ 数据集划分（train/val split）
```

**自动化特性：**
- 训练脚本自动读取配置中的 `seed` 并设置全局种子
- 数据集划分使用可复现的 Generator
- 所有随机操作都使用相同的种子

**验证方法：**
```bash
# 两次训练应产生相同结果
python main.py --exp_name CIFAR10_with_resNet34  # 第一次
python main.py --exp_name CIFAR10_with_resNet34  # 第二次（结果完全相同）
```

---

## 📊 6. 实验跟踪与可视化

### 6.1 TensorBoard（默认启用）

项目默认使用 TensorBoard 记录训练指标。日志保存在 `<output_dir>/summary/` 目录下。

**启动 TensorBoard：**
```bash
tensorboard --logdir=<output_dir>/summary
```

**记录的指标：**
- 训练/验证损失
- 准确率、F1、AUC、平衡准确率
- Cohen's Kappa、精确率、召回率、特异度

---

### 6.2 SwanLab（可选云端实验跟踪）

[SwanLab](https://swanlab.cn) 是一个强大的实验跟踪平台，支持：
- 📊 实时指标可视化
- 🖼️ 样本图像自动记录
- 🔄 K-Fold 多折实验聚合
- 🌐 云端访问和团队协作

#### ✨ 启用 SwanLab

**步骤 1：安装依赖**
```bash
pip install swanlab
```

**步骤 2：修改实验配置**

在 `configs/experiments_object.py` 中修改对应实验的配置：

```python
experiments = {
    "CIFAR10_with_resNet34": {
        # ... 其他配置 ...
        
        # ==============================================================================
        # SwanLab Configuration (Optional Experiment Tracking)
        # ==============================================================================
        "use_swanlab": True,  # ✅ 启用 SwanLab
        # "swanlab_project": "dl-classification",  # SwanLab 项目名称
        "swanlab_description": "CIFAR10 Classification with ResNet34",  # 实验描述
        "swanlab_num_samples": 8,  # 记录的样本图像数量
    }
}
```

**步骤 3：运行实验**
```bash
python main.py --exp_name CIFAR10_with_resNet34
```

#### 🎨 SwanLab 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_swanlab` | `bool` | `False` | 是否启用 SwanLab 实验跟踪 |
| `swanlab_project` | `str` | `"dl-classification"` | SwanLab 项目名称 |
| `swanlab_description` | `str` | `"Deep Learning Classification Experiment"` | 实验描述 |
| `swanlab_num_samples` | `int` | `8` | 自动记录的样本图像数量 |

#### 🔍 SwanLab 记录内容

**1. 训练指标（每个 epoch）**
- `train/loss`, `train/accuracy`, `train/f1`, `train/auc`, `train/balance_acc`
- `val/loss`, `val/accuracy`, `val/precision`, `val/recall`, `val/specificity`
- `val/f1`, `val/auc`, `val/balance_acc`, `val/cohen_kappa`

**2. K-Fold 支持**
- 所有 fold 的指标记录到**同一个 SwanLab run**
- 每个 fold 的指标带有 `fold_X/` 前缀（如 `fold_1/val/accuracy`）
- 便于对比不同 fold 的性能差异

**3. 样本图像**
- 自动从训练集提取前 N 张图像（由 `swanlab_num_samples` 指定）
- 图像会反归一化到 0-255 范围并显示标签
- **K-Fold 场景**：仅在第一个 fold 记录一次，避免重复

**4. 混淆矩阵**
- 每个 fold 的最佳模型混淆矩阵
- 归一化可视化，显示真实值 vs 预测值分布

#### 🚀 查看实验结果

运行训练后，终端会输出 SwanLab 实验链接：

```
✅ SwanLab initialized successfully. Project: dl-classification
   View experiment at: https://swanlab.cn/@username/dl-classification/runs/xxx
```

点击链接即可在浏览器中查看：
- 📈 指标曲线对比（支持多 fold 叠加显示）
- 🖼️ 样本图像展示
- 📊 混淆矩阵可视化
- ⚙️ 完整的超参数记录

#### 🔧 与 TensorBoard 共存

- SwanLab 和 TensorBoard **完全独立**，可同时启用
- SwanLab 默认**关闭**（`use_swanlab: False`），不影响现有功能
- 未安装 `swanlab` 时会优雅降级，仅打印警告而不中断训练

#### 💡 最佳实践

**单次实验：**
```python
"use_swanlab": True,
"swanlab_num_samples": 8,  # 记录 8 张样本图像
```

**K-Fold 实验：**
```python
"k_fold": 5,  # 5折交叉验证
"use_swanlab": True,  # 所有 fold 记录到同一个 run
"swanlab_num_samples": 16,  # 只在 fold 0 记录一次
```

**多实验对比：**
- 在 SwanLab 平台上同时运行多个配置
- 使用项目面板对比不同模型/超参数的效果

---

## 🧠 7. 支持的模型架构

| 模型 | 文件 | 特点 |
|------|------|------|
| **ResNet** | `ResNet.py` | 经典残差网络，稳定可靠 |
| **EfficientNet** | `EfficientNet.py` | 高效缩放，精度/速度平衡 |
| **EfficientViT** | `EfficientViT.py` | 轻量级 Vision Transformer |
| **MetaFormer** | `MetaFormer.py` | 统一 CNN/Transformer 的骨干 |
| **PoolFormer** | `PoolFormer.py` | 基于池化的纯 Transformer 替代方案 |

> 所有模型均支持 ImageNet 预训练权重加载（若可用）。

---

## 📐 8. 模型参数与 FLOPs 计算

项目提供了一个轻量工具用于计算模型参数（Params）和 FLOPs（基于 MACs）：

- 脚本路径：`utils/model_stats.py`
- 依赖：`ptflops`

示例用法：
```powershell
python -m utils.model_stats --num_classes 102 --img_size 224 --device cpu --output outputs/model_stats.txt
```

---

## ✅ 9. 模型评估（混淆矩阵可视化）

使用 `test.py` 运行模型在测试集上的评估。脚本会在指定的 `--save_dir`（脚本运行时会带时间戳生成子目录）下保存一个混淆矩阵的 PNG 图像，文件名格式类似：

```
{exp_name}_confusion_foldtest.png
```

说明：如果没有显式传入 `--exp_name` 或 observer 的 `name`，文件名前缀可能为 `None` 或 `exp`。图像默认为按真实类别行归一化的视图（显示每类的召回率分布以及每个格子的绝对样本数）。

示例运行：

```bash
python test.py --data_dir /your/image/root --test_label_file_path /path/to/test.txt \
  --checkpoint best_model.pth --batch_size 64 --num_workers 4 --num_classes 102 --save_dir ./test_outputs
```

---

## 💳 10. `experiment_results` 目录说明

用于保存每次实验的日志、模型权重和配置备份。

以`markdown`格式记录每次实验的关键指标，便于对比和复现。

目录文件说明：

```
数据集_年月日_时分秒毫秒_显卡配置
```


## 🛠️ 11. 扩展指南

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

## 📬 12. 贡献与反馈

欢迎提交 Issue 或 Pull Request！如果你觉得这个项目对你有帮助，请 ⭐ Star 支持！

---

> **Made with ❤️ and PyTorch** 
> © 2025 Deep Learning Classification Project — MIT License

---

✅ **现在就克隆项目，开启你的图像分类之旅吧！**

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=9580FF&text=✨%20Enjoy%20Building%20Your%20Model!%20✨&fontColor=FFFFFF&fontSize=25&height=80"/>
</p>