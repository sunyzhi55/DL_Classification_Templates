---
date: 2025-12-03T11:56:00  
tags:
  - python
  - deep learning
  - image classification
---





# CIFAR10 图像分类实验结果





> **📅 实验日期:** 2025-11-25 19:14 - 2025-12-03 5:38
>
> **🏷️ 任务类型:** 细粒度图像分类 (Fine-Grained Image Classification)


## 1. 实验环境 (Experimental Environment)

| 组件 | 规格/版本 | 备注 |
| :--- | :--- | :--- |
| **OS** | Rocky Linux 9.6 (Blue Onyx) (命令：`cat /etc/os-release`) | |
| **GPU** | NVIDIA RTX V100 (8 * 32GB) | CUDA 13.0 |
| **Framework** | torch  2.8.0 | |
| **Python** | 3.9.21 | |
| **主要库** | torchvision (0.23.0), scikit-learn, matplotlib | |

## 2. 数据集介绍 (Dataset Overview)

1、**数据集名称:** CIFAR10

2、**类别数量:** 10 类

3、**数据划分 (Split):**



原始训练集大小: `50000`张

原始验证集大小: `10000`张

CIFAR10数据集类别数量: `10`张

CIFAR10数据集类别名称: `['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']`





> 数据集划分：
>
> 将 原始训练集 和 原始验证集 合并得到一个新的数据集，然后进行5折交叉验证，保存每一折最好的结果。



4、**预处理与增强 (Preprocessing & Augmentation):**



```python
train_validation_test_transform={
        'train_transforms': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            # 引入RandAugment
            transforms.RandAugment(num_ops=2, magnitude=9),  # 调整 num_ops 和 magnitude 以控制强度
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # 引入RandomErasing
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')  # value='random' 使用随机像素值填充
        ]),
        'validation_transforms': transforms.Compose([
            transforms.Resize((256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'test_transforms': transforms.Compose([
            transforms.Resize((256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }
```



## 3. 模型架构 (Model Architecture)

**Backbone:**  Resnet34

**Pretrained Weights:**   无

**分类头 (Head):** `Linear(in_features=2048, out_features=100)`

**参数量 (Params) 和 计算量 (FLOPs)** 

```
Resnet34 on Oxford Flowers102
Command:
Params (raw): 21336998.0
Params (str): 21.34 M
MACs (raw): 3679558758.0
MACs (str): 3.68 GMac
Estimated FLOPs (2*MACs): 7359117516.0
```

**主要改进点:**

*在此描述你对模型做的特殊修改，例如添加了 Attention 模块，修改了 Dropout 率等。*

## 4. 训练细节、超参数

 (Training Details and Hyperparameter)



**Epoch:**  2000， **Batch Size:**  64， **Learning Rate:** 0.0001， **Weight Decay:**  0.002



**Optimizer:**

```python
AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
```

**LR Scheduler:**

```python
scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=opt.lr * 5,  # 峰值学习率（通常为初始lr的3~10倍）
            steps_per_epoch=len(train_loader),
            epochs=opt.epochs,
            anneal_strategy='cos',  # 余弦退火
            pct_start=0.1,          # 10% 的时间用于 warm-up
            # div_factor=25.0,        # 初始学习率 = max_lr / div_factor
            # final_div_factor=1e4,   # 最低学习率 = max_lr / final_div_factor
            # three_phase=False       # 可选：是否三阶段策略
        )
```

> 📌 OneCycleLR 可有效替代传统学习率衰减策略，兼顾快速收敛与稳定训练。



**Loss Function：**

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
```

> 使用 **带 Label Smoothing 的交叉熵损失（CrossEntropyLoss）**，能够缓解模型过自信、提升泛化能力，尤其在数据噪声较多或类别不均衡时效果显著。



**其他策略：**  



| 策略                  | 说明                                                         |
| :-------------------- | :----------------------------------------------------------- |
| **Early Stopping**    | (patience = 200), 监控验证指标，若连续 **200 个 Epoch** 无提升则提前停止训练，防止过拟合 |
| **Gradient Clipping** | 对梯度进行裁剪，防止梯度爆炸，提升训练稳定性                 |
| **数据加载并行**      | (num_workers = 4), 使用 **4 个 DataLoader Worker** 加速数据读取与增强 |



> **总训练时长：178.0h 24.0m 3.9902656078338623s**

## 5. 评估结果 (Evaluation Results)

### 5.1 5折交叉验证结果



| Fold     | Best Epoch | Accuracy    | Precision (micro) | Recall (micro) | Specificity | F1-score (micro) | Cohen's Kappa | Balanced Acc | AUROC       |
| -------- | ---------- | ----------- | ----------------- | -------------- | ----------- | ---------------- | ------------- | ------------ | ----------- |
| 1        | 1870       | 0.9592      | 0.9594            | 0.9589         | 0.9955      | 0.959            | 0.9546        | 0.9772       | 0.9942      |
| 2        | 610        | 0.9454      | 0.9461            | 0.9454         | 0.9939      | 0.9455           | 0.9394        | 0.9697       | 0.9929      |
| 3        | 1956       | 0.9592      | 0.9599            | 0.9595         | 0.9955      | 0.9596           | 0.9546        | 0.9775       | 0.9954      |
| 4        | 933        | 0.9486      | 0.9492            | 0.9487         | 0.9943      | 0.9488           | 0.9429        | 0.9715       | 0.9946      |
| 5        | 846        | 0.9481      | 0.9486            | 0.9481         | 0.9942      | 0.9482           | 0.9423        | 0.9712       | 0.9949      |
| **Mean** | -          | 0.9521      | 0.95264           | 0.95212        | 0.99468     | 0.95222          | 0.94676       | 0.97342      | 0.9944      |
| **Std**  | -          | 0.005898474 | 0.005819485       | 0.005889788    | 0.000682349 | 0.005889788      | 0.006509869   | 0.00326766   | 0.000846168 |





## 6. 可视化分析 (Visualization Analysis)

### 6.1 混淆矩阵 (Confusion Matrix)

5折交叉验证混淆矩阵：



Fold 1：



![CIFAR10_with_resNet34_confusion_fold1](assets/CIFAR10_with_resNet34_confusion_fold1.png)





Fold 2：

![CIFAR10_with_resNet34_confusion_fold2](assets/CIFAR10_with_resNet34_confusion_fold2.png)







Fold 3：



![CIFAR10_with_resNet34_confusion_fold3](assets/CIFAR10_with_resNet34_confusion_fold3.png)





Fold 4：

![CIFAR10_with_resNet34_confusion_fold4](assets/CIFAR10_with_resNet34_confusion_fold4.png)





Fold 5：



![CIFAR10_with_resNet34_confusion_fold5](assets/CIFAR10_with_resNet34_confusion_fold5.png)



