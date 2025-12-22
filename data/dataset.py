from pathlib import Path
from torch.utils.data import ConcatDataset
import torchvision
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
class ImageFolderDictDataset(Dataset):
    def __init__(self, img_dir, labels_file = None, transform=None, extensions=(".jpg", ".jpeg", ".png")):
        """
        一个通用的图像分类Dataset类，加载结构为：
            img_dir/
                class1/
                    xxx.jpg
                    yyy.png
                class2/
                    zzz.jpg
                    ...
        Args:
            img_dir (str or Path): 数据集根目录。
            transform (callable, optional): 图像预处理变换（如Resize、ToTensor、Normalize等）。
            extensions (tuple): 允许的图片扩展名。
        """
        self.img_dir = Path(img_dir)
        self.extensions = extensions

        # 收集类别名称（子文件夹名）
        self.classes = sorted([d.name for d in self.img_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 收集所有图片路径及对应标签
        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.img_dir / cls_name
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        if not self.samples:
            raise RuntimeError(f"未在 {self.img_dir} 下找到任何图片，请检查路径或扩展名设置。")

        # 默认 transform
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        batch = {
            "image": image,
            "label": label,
            "path": str(img_path),
            "class_name": self.classes[label]
        }
        return batch


class OxfordFlowersDataset(Dataset):
    def __init__(self, labels_file, img_dir, transform=None, target_transform=None):
        with open(labels_file, 'r') as file:
            self.img_labels = [line.strip().split() for line in file]
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir / self.img_labels[idx][0]
        image = Image.open(img_path).convert('RGB')
        label = int(self.img_labels[idx][1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        batch = {
            "image": image,
            "label": label,
            "path": str(img_path),
            "class_name": self.img_labels[idx][1]
        }
        return batch

    def count_images_per_class(self):
        # 使用defaultdict来存储每个类别的图片数量，默认值为0
        images_count = defaultdict(int)

        # 遍历所有图片标签并统计每个类别的数量
        for label in self.img_labels:
            class_label = int(label[1])  # 假设第二个元素是类别标签
            images_count[class_label] += 1

        return dict(images_count)  # 返回普通字典方便使用

# 请实现将CIFAR10中的 item 以字典返回
class CIFAR10DictDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.cifar10 = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        image, label = self.cifar10[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        batch = {
            "image": image,
            "label": label,
            "index": idx
        }
        return batch
    
def CIFAR10Dataset(img_dir, **kwargs):
    
    train_dataset = CIFAR10DictDataset(root=img_dir, train=True, transform=kwargs.get('transform', None))
    test_dataset = CIFAR10DictDataset(root=img_dir, train=False, transform=kwargs.get('transform', None))
    return {
        'train': train_dataset,
        'test': test_dataset
    }


# 请实现将CIFAR100中的 item 以字典返回
class CIFAR100DictDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.cifar100 = torchvision.datasets.CIFAR100(root=root, train=train, download=True)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.cifar100)

    def __getitem__(self, idx):
        image, label = self.cifar100[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        batch = {
            "image": image,
            "label": label,
            "index": idx
        }
        return batch
    
def CIFAR100Dataset(img_dir, **kwargs):
    train_dataset = CIFAR100DictDataset(root=img_dir, train=True, transform=kwargs.get('transform', None))
    test_dataset = CIFAR100DictDataset(root=img_dir, train=False, transform=kwargs.get('transform', None))
    return {
        'train': train_dataset,
        'test': test_dataset
    }


def get_dataset(dataset_name, **kwargs):
    """
    获取指定数据集
    """
    if dataset_name == 'OxfordFlowers':
        return OxfordFlowersDataset(**kwargs)
    elif dataset_name == 'ImageFolderDict':
        return ImageFolderDictDataset(**kwargs)
    elif dataset_name == 'CIFAR10':
        dataset = CIFAR10Dataset(**kwargs)
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        return full_dataset
    elif dataset_name == 'CIFAR100':
        dataset = CIFAR100Dataset(**kwargs)
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        return full_dataset
    else:
        raise ValueError(f"未知的数据集名称: {dataset_name}")



def get_transforms(img_size):
    """获取数据增强和预处理变换
    Args:
        img_size (int): 输入图像的目标大小
    Returns:
        dict: 包含'train_transforms'、'validation_transforms'和'test_transforms'的字典
    """
    train_validation_test_transform={
        # 'train_transforms':transforms.Compose([
        # transforms.CenterCrop(330),
        # transforms.Resize((image_size, image_size)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(45),
        # transforms.RandomAdjustSharpness(1.3, 0.5),
        # transforms.Compose([
        #     get_color_distortion(),
        #     RandomGaussianBlur(),
        # ]),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
        # ]),
        # 'validation_transforms':transforms.Compose([
        # transforms.CenterCrop(330),
        # transforms.Resize((image_size, image_size)),
        # transforms.ToTensor(),
        # # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
        # ]),
        # 'test_transforms':transforms.Compose([
        # transforms.Resize((image_size, image_size)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        # ])


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
    return train_validation_test_transform



if __name__ == "__main__":
    # 使用示例：


    # 定义图片的转换操作，例如调整大小、转换为tensor等
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 假设你的图片存储在'./images/'目录下，且你想加载训练集
    # dataset = OxfordFlowersDataset(labels_file=r"D:\BaiduNetdiskDownload\Oxford 102 Flowers\data\oxford-102-flowers\all.txt",
    #                              img_dir=r"D:\BaiduNetdiskDownload\Oxford 102 Flowers\data\oxford-102-flowers\jpg",
    #                              transform=transform)
    
    dataset = CIFAR10Dataset(img_dir=r"/data3/wangchangmiao/shenxy/PublicDataset/cifar10",
                             labels_file=None,
                             transform=transform)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建DataLoader
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    # 打印Cifar10数据集类别数量
    print(f"CIFAR10数据集类别数量: {len(train_dataset.cifar10.classes)}")
    print(f"CIFAR10数据集类别名称: {train_dataset.cifar10.classes}")
    

    # # 遍历DataLoader
    # for images, labels in data_loader:
    #     # 在这里进行模型训练或其他处理
    #     print(images.shape)
    #     print(labels.shape)
    #     pass

