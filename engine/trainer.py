import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from pathlib import Path
import swanlab
from PIL import Image
from torchvision import transforms

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, device, observer, fold=0, 
                 swanlab_config=None, full_dataset=None, img_size=224):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.observer = observer
        self.fold = fold
        self.swanlab_config = swanlab_config
        self.full_dataset = full_dataset
        self.img_size = img_size
        
        self.start_epoch = 1
        
        # 如果是第一折或单次训练，初始化SwanLab并记录样本图像
        if fold <= 1 and swanlab_config is not None:
            self._initialize_swanlab()
    
    def _initialize_swanlab(self):
        """初始化SwanLab并记录样本图像"""
        if self.swanlab_config:
            swanlab.init(**self.swanlab_config)
            print(f"✅ SwanLab initialized: {self.swanlab_config.get('experiment_name', 'experiment')}")
            
            # 记录样本图像
            if self.full_dataset is not None:
                self._log_sample_images(num_samples=20)
    
    def _log_sample_images(self, num_samples=20):
        """从数据集中提取前num_samples张图像并记录到SwanLab"""
        print(f"📸 Logging {num_samples} sample images to SwanLab...")
        
        # 创建用于可视化的transform（只做resize，不做normalize，便于显示）
        vis_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        
        sample_images = []
        num_to_fetch = min(num_samples, len(self.full_dataset))
        
        for idx in range(num_to_fetch):
            try:
                item = self.full_dataset[idx]

                
                # 处理不同的数据格式
                if isinstance(item, dict):
                    img_path = item.get('path', '')
                    label = item.get('label', 0)
                    class_name = item.get('class_name', str(label))
                    
                    # 如果有路径，从路径重新加载图像
                    if img_path and Path(img_path).exists():
                        pil_image = Image.open(img_path).convert('RGB')
                    elif 'image' in item:
                        img_data = item['image']
                        if isinstance(img_data, torch.Tensor):
                            img_np = img_data.permute(1, 2, 0).numpy()
                            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                            img_np = (img_np * 255).astype(np.uint8)
                            pil_image = Image.fromarray(img_np)
                        else:
                            pil_image = img_data
                    else:
                        continue
                else:
                    # tuple格式 (image, label)
                    pil_image, label = item
                    class_name = str(label)
                
                # 确保是PIL Image
                if not isinstance(pil_image, Image.Image):
                    if isinstance(pil_image, torch.Tensor):
                        img_np = pil_image.permute(1, 2, 0).numpy()
                        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                        img_np = (img_np * 255).astype(np.uint8)
                        pil_image = Image.fromarray(img_np)
                    elif isinstance(pil_image, np.ndarray):
                        if pil_image.max() <= 1.0:
                            pil_image = (pil_image * 255).astype(np.uint8)
                        pil_image = Image.fromarray(pil_image)
                
                # 应用可视化transform
                img_tensor = vis_transform(pil_image)
                
                sample_images.append(swanlab.Image(img_tensor, caption=f"Class: {class_name}"))
                
            except Exception as e:
                print(f"⚠️  Warning: Failed to load image {idx}: {e}")
                continue
        
        # 记录到SwanLab
        if sample_images:
            swanlab.log({"Sample_Images/Training_Samples": sample_images})
            print(f"✅ Successfully logged {len(sample_images)} images to SwanLab")
        else:
            print("⚠️  No images were logged to SwanLab")

    def train_one_epoch(self, epoch):
        self.model.train()
        self.observer.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for ii, batch in enumerate(pbar):
            image = batch.get('image')
            label = batch.get('label')
            image, label = image.to(self.device), label.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(image)
            loss = self.criterion(outputs, label)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 训练循环中
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()  # OneCycleLR在每个batch更新学习率
            
            
            prob = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(prob, dim=1)

            self.observer.train_update(loss, prob, predictions, label)

            current_lr = self.optimizer.param_groups[0]['lr']
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "LR": f"{current_lr:.6f}"
            })

    def validate(self, epoch):
        self.model.eval()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for ii, batch in enumerate(pbar):
                image = batch.get('image')
                label = batch.get('label')
                image, label = image.to(self.device), label.to(self.device)

                outputs = self.model(image)
                loss = self.criterion(outputs, label)
                
                prob = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(prob, dim=1)
                
                self.observer.eval_update(loss, prob, predictions, label)

    def run(self, epochs, is_last_fold=True):
        """训练主循环
        
        Args:
            epochs: 训练轮数
            is_last_fold: 是否是最后一折（或单次训练），用于决定是否调用swanlab.finish()
        """
        self.observer.log(f'Fold {self.fold}')
        for epoch in range(self.start_epoch, epochs + 1):
            self.train_one_epoch(epoch)
            self.validate(epoch)
            
            if self.observer.execute(epoch, epochs, len(self.train_loader.dataset), len(self.val_loader.dataset), self.fold, self.model):
                print("Early stopping triggered.")
                break
            
            # epoch结束后
            if self.scheduler is not None:
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()  # epoch级别更新
        
        self.observer.finish(self.fold)
        
        # 如果是最后一折或单次训练，结束SwanLab会话
        if is_last_fold and self.swanlab_config is not None:
            swanlab.finish()
            print("✅ SwanLab session finished")

def get_trainer(trainer_name, **kwargs):
    
    return Trainer(**kwargs)
    