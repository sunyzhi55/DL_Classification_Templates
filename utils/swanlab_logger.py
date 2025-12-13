"""
SwanLab Logger Module for Deep Learning Classification Project

This module provides SwanLab experiment tracking integration with the following features:
- Optional enabling via configuration
- K-Fold cross-validation support (all folds in one run)
- Automatic sample image logging
- Coexistence with TensorBoard
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings
from PIL import Image
from torchvision import transforms

class SwanLabLogger:
    """
    SwanLab实验跟踪日志器
    
    功能:
    - 支持可选启用/禁用
    - 自动记录训练样本图像
    - K折交叉验证支持(所有fold记录到同一个run)
    - 与TensorBoard共存
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        log_dir: str,
        enabled: bool = False,
        num_samples: int = 8,
    ):
        """
        初始化SwanLab日志器
        
        Args:
            config: 完整的实验配置字典
            log_dir: 日志保存目录
            enabled: 是否启用SwanLab (默认False保持向后兼容)
            num_samples: 要记录的样本图像数量
        """
        self.enabled = enabled
        self.num_samples = num_samples
        self.log_dir = log_dir
        self.config = config
        self.run = None
        self._swanlab_available = False
        self._images_logged = False  # 标记是否已记录图像(K折场景只记录一次)
        self.image_size = config.get("img_size", 224)
        
        if self.enabled:
            self._initialize_swanlab()
    
    def _initialize_swanlab(self):
        """初始化SwanLab (仅在启用时调用一次)"""
        try:
            import swanlab
            self._swanlab_available = True
            self.swanlab = swanlab
            
            # 初始化SwanLab run (整个实验周期只调用一次)
            self.run = swanlab.init(
                project=self.config.get('exp_name', 'dl-classification'),
                experiment_name=f"{self.config.get('exp_name', 'experiment')}_seed{self.config.get('seed', 0)}_kfold{self.config.get('fold', 0)}",
                # f"{args.exp_name}_seed{args.seed}_kfold{args.k_fold}"
                description=self.config.get('swanlab_description', 'Deep Learning Classification Experiment'),
                config=self.config,
                logdir=self.log_dir,
            )
            print(f"✅ SwanLab initialized successfully. Project: {self.config.get('swanlab_project', 'dl-classification')}")
            
        except ImportError:
            warnings.warn(
                "SwanLab is enabled in config but 'swanlab' package is not installed. "
                "Install it with: pip install swanlab\n"
                "SwanLab logging will be disabled.",
                UserWarning
            )
            self.enabled = False
            self._swanlab_available = False
        except Exception as e:
            warnings.warn(f"Failed to initialize SwanLab: {e}\nSwanLab logging will be disabled.", UserWarning)
            self.enabled = False
            self._swanlab_available = False

    def log_sample_images(self, dataset, fold: int = 0):
        """
        从数据集中记录样本图像到SwanLab
        
        Args:
            dataset: PyTorch Dataset对象
            fold: 当前fold编号 (K折场景下，仅在fold=0时记录)
        """
        if not self.enabled or not self._swanlab_available:
            return
        
        # K折场景：只在第一个fold记录图像，避免重复
        if self._images_logged:
            return
        import swanlab
        print(f"📸 Logging {self.num_samples} sample images to SwanLab...")
        
        # 创建用于可视化的transform（只做resize，不做normalize，便于显示）
        vis_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        sample_images = []
        num_to_fetch = min(self.num_samples, len(dataset))
        
        for idx in range(num_to_fetch):
            try:
                item = dataset[idx]
                
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
                            img_np = img_data.permute(1, 2, 0).cpu().numpy()
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
                        img_np = pil_image.permute(1, 2, 0).cpu().numpy()
                        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                        img_np = (img_np * 255).astype(np.uint8)
                        pil_image = Image.fromarray(img_np)
                    elif isinstance(pil_image, np.ndarray):
                        if pil_image.max() <= 1.0:
                            pil_image = (pil_image * 255).astype(np.uint8)
                        pil_image = Image.fromarray(pil_image)
                
                # 应用可视化transform
                img_tensor = vis_transform(pil_image)
                
                sample_images.append(swanlab.Image(img_tensor, caption=f"Sample {idx+1} - Label: {label}"))
                
            except Exception as e:
                print(f"⚠️  Warning: Failed to load image {idx}: {e}")
                continue
        
        # 记录到SwanLab
        if sample_images:
            swanlab.log({"Sample_Images/Training_Samples": sample_images})
            print(f"✅ Successfully logged {len(sample_images)} images to SwanLab")
            self._images_logged = True
        else:
            print("⚠️  No images were logged to SwanLab")


    def log_metrics(self, metrics: Dict[str, float], step: int, fold: Optional[int] = None):
        """
        记录训练/验证指标到SwanLab
        
        Args:
            metrics: 指标字典 (例如 {'train/loss': 0.5, 'val/acc': 0.9})
            step: 当前步数(epoch)
            fold: fold编号 (K折时使用，会添加fold前缀)
        """
        if not self.enabled or not self._swanlab_available:
            return
        
        try:
            # K折场景：为每个指标添加fold前缀
            if fold is not None and fold > 0:
                metrics_with_fold = {f"fold_{fold}/{k}": v for k, v in metrics.items()}
            else:
                metrics_with_fold = metrics
            
            self.swanlab.log(metrics_with_fold, step=step)
            
        except Exception as e:
            warnings.warn(f"Failed to log metrics to SwanLab: {e}", UserWarning)
    
    def finish(self):
        """完成SwanLab实验记录"""
        if self.enabled and self._swanlab_available and self.run is not None:
            try:
                self.swanlab.finish()
                print("✅ SwanLab experiment finished successfully")
            except Exception as e:
                warnings.warn(f"Error finishing SwanLab run: {e}", UserWarning)
    
    def __del__(self):
        """析构函数，确保SwanLab正确关闭"""
        # 注意：在K折场景中，只在最后一个fold完成后才调用finish
        pass


def create_swanlab_logger(config: Dict[str, Any], log_dir: str) -> SwanLabLogger:
    """
    工厂函数：根据配置创建SwanLab日志器
    
    Args:
        config: 实验配置字典
        log_dir: 日志目录
        experiment_name: 实验名称
    
    Returns:
        SwanLabLogger实例
    """
    enabled = config.get('use_swanlab', False)
    num_samples = config.get('swanlab_num_samples', 8)
    
    return SwanLabLogger(
        config=config,
        log_dir=log_dir,
        enabled=enabled,
        num_samples=num_samples
    )
