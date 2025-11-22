---
date: 2024-02-06T11:51:00
tags:
  - python
  - function
  - JS
---



da

# 临时函数





## 1、测试`nn.BCEWithLogitsLoss`函数

### 1.1 代码

```python

"""
测试BCEWithLogitsLoss函数
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def validate_loss(output, target, weight=None, pos_weight=None):
    output = F.sigmoid(output)
    # 处理正负样本不均衡问题
    if pos_weight is None:
        label_size = output.size()[1]
        pos_weight = torch.ones(label_size)
    # 处理多标签不平衡问题
    if weight is None:
        label_size = output.size()[1]
        weight = torch.ones(label_size)

    val = 0
    for li_x, li_y in zip(output, target):
        for i, xy in enumerate(zip(li_x, li_y)):
            x, y = xy
            loss_val = pos_weight[i] * y * math.log(x, math.e) + (1 - y) * math.log(1 - x, math.e)
            val += weight[i] * loss_val
    return -val / (output.size()[0] * output.size(1))


weight = torch.Tensor([0.8, 1, 0.8])
loss = nn.MultiLabelSoftMarginLoss(weight=weight)

x = torch.Tensor([[0.8, 0.9, 0.3], [0.8, 0.9, 0.3], [0.8, 0.9, 0.3], [0.8, 0.9, 0.3]])
y = torch.Tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
print(x.size())
print(y.size())
loss_val = loss(x, y)
print(loss_val.item())

validate_loss = validate_loss(x, y, weight=weight)
print(validate_loss.item())

loss = torch.nn.BCEWithLogitsLoss(weight=weight)
loss_val = loss(x, y)
print(loss_val.item())


```



### 1.2 输出结果

```
torch.Size([4, 3])
torch.Size([4, 3])
0.4405062198638916torch.Size([4, 3])
torch.Size([4, 3])
0.4405062198638916
0.4405062198638916
0.4405062198638916
0.4405062198638916
0.4405062198638916
```



## 2、统计`csv`中的类别数量

### 2.1 统计1的个数

#### 2.1.1 代码



```python
"""
已知有一个csv文件，用python实现：
第一列是文件名，之后8列是标签，每一列的值都是0或者1，请统计每个标签中1的个数，算出比例
注意：使用pathlib代替os，并且封装成一个函数进行调用
"""
import pandas as pd
from pathlib import Path


def analyze_labels(csv_path):
    """
    分析给定CSV文件中的标签分布情况。

    参数:
        csv_path (str or Path): CSV文件的路径。

    返回:
        pd.DataFrame: 包含每个标签的数量和比例。
    """
    # 使用Path处理路径
    csv_path = Path(csv_path)

    # 读取CSV文件
    df = pd.read_csv(csv_path)
    # print("df", df)

    # 假设从第二列到第九列为标签列
    label_columns = df.columns[1:9]

    # 创建一个字典存储每个标签的数量和比例
    stats = {}

    total_rows = df.shape[0]

    for col in label_columns:
        count = df[col].sum()
        ratio = 10 * count / total_rows
        stats[col] = {'Count': count, 'Ratio': ratio}

    # 将统计结果转换为DataFrame以便查看
    stats_df = pd.DataFrame.from_dict(stats, orient='index')

    return stats_df


# 示例调用
csv_file_path = r"D:\BaiduNetdiskDownload\服务外包\csv\Left_Fundus_Classification.csv"  # 替换为你的CSV文件路径
result = analyze_labels(csv_file_path)
print(result)
```



#### 2.1.2 输出结果

```
   Count     Ratio
N   1464  5.074523
D    364  1.261698
G    180  0.623917
C    159  0.551127
A    135  0.467938
H     72  0.249567
M    116  0.402080
O    615  2.131716
```

> [!important]
>
> 





### 2.2 统计0-1个数

#### 2.2.1 代码



```python
import pandas as pd
from pathlib import Path
"""
已知有一个csv文件，用python实现：
第一列是文件名，之后8列是标签，每一列的值都是0或者1，请统计每个标签的正类和负类的比例，即0的个数：1的个数，算出比例
注意：使用pathlib代替os，并且封装成一个函数进行调用
"""

def calculate_label_ratios(csv_file_path: str):
    # 创建 Path 对象来处理文件路径
    file_path = Path(csv_file_path)

    if not file_path.exists():
        print(f"文件 {file_path} 不存在！")
        return

    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 获取标签列（假设标签从第二列开始，即索引1到8）
    label_columns = df.columns[1:9]

    # 计算每一列的0和1的数量
    for label in label_columns:
        label_0_count = (df[label] == 0).sum()  # 统计值为0的数量
        label_1_count = (df[label] == 1).sum()  # 统计值为1的数量
        if label_1_count == 0:  # 防止除以0
            ratio = float('inf')  # 如果 1 的个数是 0，可以设为无穷大或其他标志
        else:
            ratio = label_0_count / label_1_count  # 计算0的个数与1的个数的比例
        print(f"{label} - 0: {label_0_count}, 1: {label_1_count}, ratio (0/1) = {ratio:.2f}")


# 使用示例
calculate_label_ratios(r"D:\BaiduNetdiskDownload\服务外包\csv\total_valid.csv")
```

#### 2.2.2 输出结果

```
N - 0: 2858, 1: 2904, ratio (0/1) = 0.98
D - 0: 5032, 1: 730, ratio (0/1) = 6.89
G - 0: 5433, 1: 329, ratio (0/1) = 16.51
C - 0: 5450, 1: 312, ratio (0/1) = 17.47
A - 0: 5486, 1: 276, ratio (0/1) = 19.88
H - 0: 5617, 1: 145, ratio (0/1) = 38.74
M - 0: 5518, 1: 244, ratio (0/1) = 22.61
O - 0: 4497, 1: 1265, ratio (0/1) = 3.55
```



## 3 合并两个`csv`文件

### 3.1 普通合并



```python
import pandas as pd

# 读取第一个CSV文件
df1 = pd.read_csv('file1.csv')

# 读取第二个CSV文件
df2 = pd.read_csv('file2.csv')

# 按行合并两个DataFrame
combined_df = pd.concat([df1, df2], ignore_index=True)

# 将合并后的DataFrame写入一个新的CSV文件
combined_df.to_csv('combined_file.csv', index=False)
```

### 3.2  交叉合并

```python
import pandas as pd

# 读取第一个CSV文件
df1 = pd.read_csv(r"C:\Users\y8549\Desktop\confuse\服务外包\Left_Fundus_Classification.csv")

# 读取第二个CSV文件
df2 = pd.read_csv(r"C:\Users\y8549\Desktop\confuse\服务外包\Right_Fundus_Classification.csv")



combined_records = []

# 获取两个DataFrame的长度
len1, len2 = len(df1), len(df2)

# 计算需要迭代的最大次数
max_iter = max(len1, len2)

# 交替添加记录到合并后的列表中
for i in range(max_iter):
    if i < len1:
        combined_records.append(df1.iloc[i].to_dict())
    if i < len2:
        combined_records.append(df2.iloc[i].to_dict())

# 将合并后的记录转换为DataFrame
combined_df = pd.DataFrame(combined_records)

# 将合并后的DataFrame写入一个新的CSV文件
combined_df.to_csv('cross_combined_file.csv', index=False)
```

## 4 图片噪声

### 4.1 往图片中加入噪声

#### 4.1.1 代码

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

# 定义一个函数来读取图片并转换为PyTorch张量
def read_image(image_path):
    # 打开图片并转换为RGB模式
    image = Image.open(image_path).convert('RGB')
    # 定义一个转换，将PIL图像转换为PyTorch张量，并归一化到[0, 1]范围
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    # 应用转换
    tensor_image = transform(image).unsqueeze(0)  # 添加批处理维度
    return tensor_image


# 定义一个函数来将噪声逐步加入到图片中
def add_noise_to_image(image_tensor, noise_tensor, num_iterations):
    # 确保图像张量和噪声张量在相同的设备上（CPU或GPU）
    device = image_tensor.device
    noise_tensor = noise_tensor.to(device)

    # 创建一个副本用于存储加噪后的图像
    noisy_image_tensor = image_tensor.clone()

    # 循环加入噪声
    for _ in range(num_iterations):
        # 将噪声加入到图像中（这里简单地使用加法，但你可以根据需要调整）
        noisy_image_tensor += 0.2 * noise_tensor

        # 确保像素值在[0, 1]范围内（由于噪声的加入，值可能会超出这个范围）
        noisy_image_tensor = torch.clamp(noisy_image_tensor, 0, 1)

    return noisy_image_tensor

if __name__ == "__main__":
    # 图片路径
    image_path = r"C:\Users\y8549\Pictures\Saved Pictures\picture.jpg" # 替换为你的图片路径
    # 读取图片并转换为张量
    image_tensor = read_image(image_path)

    # 创建一个随机噪声张量
    noise_tensor = torch.randn(1, 3, 256, 256)

    # 选择循环的次数
    num_iterations = 1

    # 将噪声加入到图片中
    noisy_image_tensor = add_noise_to_image(image_tensor, noise_tensor, num_iterations)

    # 将加噪后的图像张量转换回PIL图像并保存
    transform_to_pil = transforms.ToPILImage()
    noisy_image = transform_to_pil(noisy_image_tensor.squeeze(0))  # 移除批处理维度
    noisy_image.save('noisy_image.jpg')

    # 显示加噪后的图像（可选）
    noisy_image.show()


```

> 说明：
>
> 使用循环每次向图片中加入噪声。

### 4.2 随机生成噪声图片

#### 4.2.1 代码

```python
import torch
import numpy as np
from PIL import Image

# 创建一个随机噪声张量
noise = torch.randn(1, 3, 256, 256)

# 将张量从范围[-inf, inf]归一化到[0, 1]并转换为numpy数组
noise = noise.clamp_(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()

# 将numpy数组的值从[0, 1]缩放到[0, 255]并转换为无符号8位整数
noise_image = (noise * 255).astype(np.uint8)

# 将numpy数组转换为PIL图像并保存
image = Image.fromarray(noise_image, 'RGB')
image.save('noise_image.png')
```

## 5 Grad-CAM图

### 5.1 代码

```python
import cv2
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        print("output", output.shape)  # output torch.Size([1, 7])
        # output = output[2]  # 注意：如果模型是多输出，需要选择自己想要的输出
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h + size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w + size]

    return img


if __name__ == '__main__':
    model = poolformer_s12(num_classes=1000) # 导入自己的模型
    model.head = torch.nn.Linear(model.head.in_features, 7)
    model.load_state_dict(torch.load(r"D:\BaiduNetdiskDownload\服务外包\poolformers12_model_fold3.pth",
                                     weights_only=True, map_location=torch.device('cpu'))) # 加载自己模型的训练权重
    model.eval()  # 设置为评估模式

    # target_layers = [model.backbone.layer4]
    target_layers = [model.network[-1][-1].mlp]
    print("target_layers: ", target_layers)
    """
    target_layers:  [Mlp(
          (fc1): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))
          (act): GELU(approximate='none')
          (fc2): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
          (drop): Dropout(p=0.0, inplace=False)
        )]
    """

    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = r"D:\BaiduNetdiskDownload\服务外包\Data\Enhanced\1_left.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 448)  # 将导入的图片reshape到自己想要的尺寸

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    # target_category = 281  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    target_category = 4

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)

    output_file = 'superimposed_grad_cam.png'
    plt.imsave(output_file, visualization / 255.0)  # 注意：imsave期望输入在0-1范围内，所以再次归一化
    print(f"Superimposed image saved to {output_file}")
    # plt.imshow(visualization)
    # plt.show()


```

## 6 导出1个 2个 多个疾病的csv

```python
import pandas as pd

if __name__ == '__main__':
    # 读取CSV文件
    df = pd.read_csv(r"C:\Users\y8549\Desktop\confuse\外包数据集\Left_Fundus_Classification.csv")

    prefix_path = r"C:\Users\y8549\Desktop\confuse\外包数据集"

    # 假设第一列是名字，第二列是正常，接下来的7列是疾病数据
    # 提取正常列和疾病列
    normal_and_disease_columns = df.columns[1:9]  # 第二列到第九列

    # 检查是否所有相关列全为 0
    invalid_data = df[df[normal_and_disease_columns].sum(axis=1) == 0]
    if not invalid_data.empty:
        invalid_data.to_csv(rf"{prefix_path}\Left_invalid_data.csv", index=False)
        print(f"已导出：invalid_data.csv, 记录数:{len(invalid_data)}")

    # 提取疾病列（从第三列到第九列）
    disease_columns = df.columns[2:]

    # 创建一个新的列，计算每行的疾病数量
    df['disease_count'] = df[disease_columns].sum(axis=1)

    # 筛选“无疾病或只有一个疾病”的记录
    group_0_or_1 = df[(df['disease_count'] <= 1) & (df[normal_and_disease_columns].sum(axis=1) > 0)]
    if not group_0_or_1.empty:
        group_0_or_1.to_csv(fr"{prefix_path}\Left_group_0_or_1_diseases.csv", index=False)
        print(f"已导出：group_0_or_1_diseases.csv, 记录数:{len(group_0_or_1)}")

    # 筛选“有两个及以上疾病”的记录，并分别导出
    for count in range(2, 9):  # 疾病数量从2到8
        group_df = df[(df['disease_count'] == count) & (df[normal_and_disease_columns].sum(axis=1) > 0)]
        if not group_df.empty:  # 仅在有记录时导出
            filename = fr"{prefix_path}\Left_group_{count}_diseases.csv"
            group_df.to_csv(filename, index=False)
            print(f"已导出：{filename}, 记录数:{len(group_df)}")

```

## 7 训练集 测试集 验证集的划分

```python
import shutil
import random
from pathlib import Path

def split_images(base_dir: Path, output_dir: Path, split_ratio=(0.7, 0.2, 0.1)):
    """
    将图片按照指定的比例划分到训练集、验证集和测试集中。

    参数:
    base_dir (Path): 包含所有子文件夹（类别）的原始文件夹路径。
    output_dir (Path): 输出文件夹路径，其中将创建train、validation和test子文件夹。
    split_ratio (tuple): 一个包含三个浮点数的元组，表示训练集、验证集和测试集的比例（默认为7:2:1）。

    返回:
    None
    """
    # 确保输出文件夹及其子文件夹存在
    (output_dir / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'validation').mkdir(parents=True, exist_ok=True)
    (output_dir / 'test').mkdir(parents=True, exist_ok=True)

    # 获取所有子文件夹名称（类别）
    categories = [d.name for d in base_dir.iterdir() if d.is_dir()]

    # 遍历每个类别
    for category in categories:
        category_path = base_dir / category
        images = [f for f in category_path.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')]

        # 确保有足够的图片进行划分
        # if len(images) < sum(int(ratio * len(images) + 0.5) for ratio in split_ratio):
        #     print(f"警告：类别 '{category}' 中的图片数量不足以进行划分。")
        #     continue

        # 打乱图片顺序
        random.shuffle(images)

        # 计算每个集合的图片数量
        train_size = int(split_ratio[0] * len(images))
        validation_size = int(split_ratio[1] * len(images))
        test_size = len(images) - train_size - validation_size

        # 划分图片
        train_images = images[:train_size]
        validation_images = images[train_size:train_size + validation_size]
        test_images = images[train_size + validation_size:]

        # 创建每个类别在目标文件夹中的子文件夹
        (output_dir / 'train' / category).mkdir(parents=True, exist_ok=True)
        (output_dir / 'validation' / category).mkdir(parents=True, exist_ok=True)
        (output_dir / 'test' / category).mkdir(parents=True, exist_ok=True)

        # 复制图片到目标文件夹
        for img in train_images:
            shutil.copy(img, output_dir / 'train' / category)
        for img in validation_images:
            shutil.copy(img, output_dir / 'validation' / category)
        for img in test_images:
            shutil.copy(img, output_dir / 'test' / category)

    print("图片划分完成！")


if __name__ == "__main__":
    # 使用示例
    base_directory = Path('path/to/your/main/folder')
    output_directory = Path('path/to/output/folder')
    split_images(base_directory, output_directory)

```

## 8 生成csv文件

### （1）找出所有不成对的图片

请用python实现，一个文件夹下面有很多眼底图片，有些是成对的，有些不是，成对的图片数字一样，如0_left 和 0_right，其中0就是id，请读取图片，找出所有不成对的图片将图片导出到一个csv文件中，每一行有两列，对应一个人的左眼和右眼图片名称，如果不成对，则显示一列即可

```python
from pathlib import Path
import csv


def export_eye_images_to_csv(folder_path, output_csv):
        # 创建 Path 对象
        folder = Path(folder_path)

        # 获取所有图片文件名
        image_files = [f.name for f in folder.iterdir() if
                       f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]

        # 用于存储成对和不成对的图片
        paired_images = []
        unpaired_images = set()

        # 遍历图片文件名，提取数字部分并分类
        for img in image_files:
                name, ext = img.split('.')
                parts = name.split('_')
                if len(parts) == 2:  # 确保格式为 "数字_left/right"
                        number, side = parts
                        if side in ['left', 'right']:
                                if (number, 'left') in unpaired_images and side == 'right':
                                        # 找到成对的右眼图片
                                        paired_images.append((f"{number}_left.{ext}", f"{number}_right.{ext}"))
                                        unpaired_images.remove((number, 'left'))
                                elif (number, 'right') in unpaired_images and side == 'left':
                                        # 找到成对的左眼图片
                                        paired_images.append((f"{number}_left.{ext}", f"{number}_right.{ext}"))
                                        unpaired_images.remove((number, 'right'))
                                else:
                                        # 添加到未成对集合中
                                        unpaired_images.add((number, side))
                        else:
                                # 不符合 left/right 格式的图片
                                unpaired_images.add((name, ''))
                else:
                        # 不符合格式的图片
                        unpaired_images.add((name, ''))

        # 将未成对的图片单独列出
        unpaired_list = [f"{name}.{ext}" for name, ext in unpaired_images]

        # 写入 CSV 文件
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # 写入成对的图片
                for left, right in paired_images:
                        writer.writerow([left, right])

                # 写入未成对的图片
                for img in unpaired_list:
                        writer.writerow([img])


# 示例用法
folder_path = './eye_images'  # 替换为你的图片文件夹路径
output_csv = './output.csv'  # 输出的 CSV 文件路径
export_eye_images_to_csv(folder_path, output_csv)
```

### （2）导出新的csv文件

请用python实现，一个文件夹下面有很多眼底图片，有些是成对的，有些不是，成对的图片数字一样，如0_left 和 0_right，其中0就是id
请读取图片，找出所有不成对的图片，同时有一个csv文件，第一列对应每个成对的图片的id，请基于这个csv文件，将所有不成对的id的对应的行删除，文件夹中没有的，但是csv中也有，这个csv中也要删除
导出到新的csv文件

```python
import os
import csv

def find_valid_ids(folder_path):
    """
    找出文件夹中所有成对的图片 ID。
    成对的图片格式为 {id}_left 和 {id}_right。
    """
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    ids = {}

    # 遍历所有图片文件，提取 id 并记录 left 和 right 的存在情况
    for file_name in image_files:
        name, ext = os.path.splitext(file_name)
        if '_' in name:
            id_part, side = name.rsplit('_', 1)
            if id_part not in ids:
                ids[id_part] = {'left': False, 'right': False}
            if side == 'left':
                ids[id_part]['left'] = True
            elif side == 'right':
                ids[id_part]['right'] = True

    # 找出成对的 id
    valid_ids = []
    for id_part, sides in ids.items():
        if sides['left'] and sides['right']:
            valid_ids.append(id_part)

    return valid_ids


def filter_csv_by_valid_ids(csv_file, output_csv, valid_ids):
    """
    根据有效的 id 保留 CSV 文件中的对应行。
    如果某个 id 不在有效 id 列表中，则删除该行。
    """
    with open(csv_file, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 写入表头
        header = next(reader)
        writer.writerow(header)

        # 只保留有效 id 对应的行
        for row in reader:
            id_part = row[0]  # 假设第一列是 id
            if id_part in valid_ids:
                writer.writerow(row)


if __name__ == "__main__":
    # 输入文件夹路径和 CSV 文件路径
    folder_path = "path/to/your/image/folder"  # 替换为你的图片文件夹路径
    csv_file = "path/to/your/input.csv"        # 替换为你的输入 CSV 文件路径
    output_csv = "path/to/your/output.csv"     # 替换为你的输出 CSV 文件路径

    # 找出文件夹中成对的图片 ID
    valid_ids = find_valid_ids(folder_path)
    print(f"成对的图片 ID: {valid_ids}")

    # 根据成对的 ID 过滤 CSV 文件
    filter_csv_by_valid_ids(csv_file, output_csv, valid_ids)
    print(f"已生成过滤后的 CSV 文件: {output_csv}")
```



dasd按时

等等s

大三



das>

> [!note]
>
> 

> [!important]
>
> 

> [!tip]
>
> 

> [!warning]
>
> 

> [!caution]
>
> 









