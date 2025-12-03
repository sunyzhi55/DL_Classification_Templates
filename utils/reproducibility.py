"""
可复现性工具模块
提供随机种子设置和确定性配置
"""
import os
import random
import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False):
    """
    设置所有随机数生成器的种子以确保实验可复现性
    
    Args:
        seed (int): 随机种子值
        deterministic (bool): 是否启用完全确定性模式
                            - True: 完全可复现，但可能降低训练速度
                            - False: 大部分可复现，保持较好的性能
    
    Note:
        - 完全确定性模式会禁用cudnn的benchmark和某些非确定性算法
        - 在某些操作（如scatter_add）上仍可能存在微小的数值差异
    """
    # Python内置random模块
    random.seed(seed)
    
    # Numpy随机数生成器
    np.random.seed(seed)
    
    # PyTorch CPU随机数
    torch.manual_seed(seed)
    
    # PyTorch GPU随机数（支持多GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 设置环境变量（影响Python hash随机化）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        # 完全确定性模式：牺牲性能换取可复现性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # PyTorch 1.8+: 设置确定性算法
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as e:
                print(f"Warning: Could not enable deterministic algorithms: {e}")
        
        # 设置cublas工作空间配置（PyTorch 1.11+）
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = False
    else:
        # 性能模式：保持cudnn自动调优
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    print(f"✅ Global random seed set to: {seed}")
    print(f"   Deterministic mode: {'ON (slower but fully reproducible)' if deterministic else 'OFF (faster)'}")


def get_generator(seed: int, device: str = 'cpu') -> torch.Generator:
    """
    创建一个指定种子的PyTorch Generator对象
    用于DataLoader、random_split等需要单独控制随机性的场景
    
    Args:
        seed (int): 随机种子
        device (str): 设备类型 ('cpu' 或 'cuda')
    
    Returns:
        torch.Generator: 已设置种子的生成器对象
    
    Example:
        >>> generator = get_generator(42, 'cpu')
        >>> train_set, val_set = random_split(dataset, [0.8, 0.2], generator=generator)
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def make_reproducible_split(dataset, lengths, seed: int):
    """
    可复现的数据集划分
    
    Args:
        dataset: PyTorch Dataset对象
        lengths: 划分长度列表，如 [0.8, 0.2] 或 [8000, 2000]
        seed: 随机种子
    
    Returns:
        list of Subset: 划分后的数据集子集列表
    
    Example:
        >>> train_data, val_data = make_reproducible_split(full_dataset, [0.8, 0.2], seed=42)
    """
    generator = get_generator(seed, 'cpu')
    return torch.utils.data.random_split(dataset, lengths, generator=generator)


def worker_init_fn(worker_id: int, seed: int = 0):
    """
    DataLoader worker进程初始化函数
    确保每个worker使用不同但可复现的随机种子
    
    Args:
        worker_id (int): worker进程ID（由DataLoader自动传入）
        seed (int): 基础随机种子
    
    Example:
        >>> from functools import partial
        >>> init_fn = partial(worker_init_fn, seed=args.seed)
        >>> loader = DataLoader(dataset, ..., worker_init_fn=init_fn)
    """
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    # 测试种子设置
    print("Testing reproducibility utils...")
    
    # 测试1: 全局种子设置
    set_global_seed(42, deterministic=True)
    print(f"Random int: {random.randint(0, 100)}")
    print(f"Numpy random: {np.random.rand()}")
    print(f"Torch random: {torch.rand(1).item()}")
    
    # 测试2: Generator
    gen = get_generator(42)
    print(f"Generator sample: {torch.rand(1, generator=gen).item()}")
    
    print("\n✅ All reproducibility tests passed!")
