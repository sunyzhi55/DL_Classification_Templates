import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from sklearn.model_selection import KFold
from configs.config import get_config
from data.dataset import  get_transforms, get_dataset
from engine.trainer import Trainer, get_trainer
from pathlib import Path
from models.get_model import get_model
from utils.basic import get_optimizer, get_scheduler
from utils.loss_function import get_loss_function
from utils.observer import RuntimeObserver
from utils.reproducibility import set_global_seed, make_reproducible_split
import copy
# Wrapper to apply transforms dynamically

class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        item = self.subset[index]
        # 统一处理：支持dict和tuple两种格式
        if isinstance(item, dict):
            image, label = item['image'], item['label']
            if self.transform:
                image = self.transform(image)
            return {'image': image, 'label': label, **{k:v for k,v in item.items() if k not in ['image','label']}}
        else:
            image, label = item
            if self.transform:
                image = self.transform(image)
            return {'image': image, 'label': label}

    def __len__(self):
        return len(self.subset)


if __name__ == '__main__':
    args = get_config()
    
    # ========== 设置全局随机种子确保可复现性 ==========
    set_global_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset
    train_transform = get_transforms(args.img_size)['train_transforms']
    val_transform = get_transforms(args.img_size)['validation_transforms']
    test_transform = get_transforms(args.img_size)['test_transforms']
    
    # Use dataset class from config
    full_dataset = get_dataset(args.dataset_name, img_dir=args.data_dir,
                               labels_file=args.train_eval_label_file_path,
                               transform=None)
    print(f"Total images in full dataset: {len(full_dataset)}")
    
    
    # 2. Training Loop (K-Fold or Single Split)
    start = time.time()
    if args.k_fold > 1:
        print(f"Starting {args.k_fold}-Fold Cross Validation")
        kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
        
        best_train_eval_dict = {}
        test_dict = {}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
            print(f"\n============================")
            print(f"🔥 开始训练 Fold {fold + 1} / {args.k_fold}")
            print("============================\n")
            
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)
            
            train_data = TransformSubset(train_subset, transform=train_transform)
            val_data = TransformSubset(val_subset, transform=val_transform)
            
            # main.py
            train_loader = DataLoader(
                train_data, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.num_workers,
                pin_memory=True,           # ✅ 加速GPU传输
                prefetch_factor=2,         # ✅ 预取数据
                persistent_workers=True    # ✅ 保持worker进程
            )
            val_loader = DataLoader(
                val_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            
            # Model
            model = get_model(args.model_name, args.num_classes, args.checkpoint_path, device)
            
                
            # model = model.to(device)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model initialized with {total_params} parameters.")

            # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
            # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            optimizer = get_optimizer(args.optimizer_name, model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            
            # args.scheduler is likely the get_scheduler function itself
            scheduler = get_scheduler(optimizer, args, train_loader)
            
            # criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
            # criterion = args.criterion(label_smoothing=args.label_smoothing).to(device)
            criterion = get_loss_function(args.loss_fn_name, device, label_smoothing=args.label_smoothing)
            
            # Observer
            observer = RuntimeObserver(
                log_dir=args.save_dir,
                device=device,
                num_classes=args.num_classes,
                task="multiclass" if args.num_classes > 2 else "binary",
                average='macro' if args.num_classes > 2 else 'micro',
                patience=args.patience,
                name=args.exp_name,
                seed=args.seed,
                hyperparameters=vars(args)  # 传递所有超参数
            )
            trainer = get_trainer(args.trainer_name, model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, scheduler=scheduler, criterion=criterion, device=device, observer=observer, fold=fold+1)
            trainer.run(args.epochs)
            best_train_eval_dict[fold+1] = copy.deepcopy(observer.best_dicts)
        
        # if test_dataset is not None:
        #     # ... (omitted code)
        
        print("\nK-Fold Cross Validation Complete.")
        print("Best Results per Fold:")
        for fold, results in best_train_eval_dict.items():
            print(f"Fold {fold}: {results}")
        
    else:
        # Simple split 80/20 with reproducible seed
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        # 使用可复现的数据集划分
        train_subset, val_subset = make_reproducible_split(full_dataset, [train_size, val_size], seed=args.seed)
        
        train_data = TransformSubset(train_subset, transform=train_transform)
        val_data = TransformSubset(val_subset, transform=val_transform)
            
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # model = get_model(args.num_classes, args.checkpoint_path, device)
        model = get_model(args.model_name, args.num_classes, args.checkpoint_path, device)

        # model = model.to(device)
        
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = get_optimizer(args.optimizer_name, model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # scheduler = get_scheduler(optimizer, args, train_loader)
        scheduler = get_scheduler(optimizer, args, train_loader)
        
        # criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        criterion = get_loss_function(args.loss_fn_name, device, label_smoothing=args.label_smoothing)
        
        # Observer
        print(f"args.save_dir: {args.save_dir}")
        observer = RuntimeObserver(
            log_dir=str(Path(args.save_dir)) + '/',
            device=device,
            num_classes=args.num_classes,
            task="multiclass" if args.num_classes > 2 else "binary",
            average='macro' if args.num_classes > 2 else 'micro',
            patience=args.patience,
            name=args.exp_name,
            seed=args.seed,
            hyperparameters=vars(args)  # 传递所有超参数
        )
        trainer = get_trainer(
            args.trainer_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            observer=observer,
            fold=0
        )
        trainer.run(args.epochs)
    end = time.time()
    total_seconds = end - start
    # 计算小时、分钟和秒
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    # 打印总训练时间
    print(f"Total training time: {hours}h {minutes}m {seconds}s")