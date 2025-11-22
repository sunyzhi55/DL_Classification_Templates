import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from sklearn.model_selection import KFold
from configs.config import get_config
from data.dataset import  get_transforms, OxfordFlowersDataset
from engine.trainer import Trainer
from pathlib import Path
from utils.basic import get_scheduler
from utils.observer import RuntimeObserver
from models.get_model import get_model
import copy
# Wrapper to apply transforms dynamically

class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        sample = self.subset[index]
        if self.transform:
            # 对MRI和PET分别应用转换
            if 'image' in sample:
                sample['image'] = self.transform(sample['image'])
                sample['image'] = sample['image']
        return sample

    def __len__(self):
        return len(self.subset)


if __name__ == '__main__':
    args = get_config()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset
    train_transform = get_transforms(args.img_size)['train_transforms']
    val_transform = get_transforms(args.img_size)['validation_transforms']
    test_transform = get_transforms(args.img_size)['test_transforms']
    full_dataset = OxfordFlowersDataset(labels_file=args.train_eval_label_file_path, img_dir=args.data_dir, transform=None)
    test_dataset = OxfordFlowersDataset(labels_file=args.test_label_file_path, img_dir=args.data_dir, transform=None)
    
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
            
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
            # Model
            model = get_model(args.num_classes, args.checkpoint_path, device)
            model = model.to(device)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model resnet34 initialized with {total_params} parameters.")

            # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = get_scheduler(optimizer, args, train_loader)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
            
            # Observer
            observer = RuntimeObserver(
                log_dir=args.save_dir,
                device=device,
                num_classes=args.num_classes,
                task="multiclass" if args.num_classes > 2 else "binary",
                average='macro' if args.num_classes > 2 else 'micro',
                patience=args.patience,
                name=args.exp_name,
                seed=args.seed
            )
            
            trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, criterion, device, observer, fold=fold+1)
            trainer.run(args.epochs)
            best_train_eval_dict[fold+1] = copy.deepcopy(observer.best_dicts)
        
        # if test_dataset is not None:
        #     # 在每折结束后使用最佳模型对测试集做一次完整评估并记录
        #     test_save_path = Path(best_result_model_path) / f"{args.exp_name}_best_model_fold{fold}.pth"
        #     test_ids = TransformedSubset(test_dataset, transform=total_transform['test_transforms'])
        #     test_dataloader = DataLoader(dataset=test_ids, batch_size=batch_size, 
        #                                  shuffle=False, num_workers=workers)
        #     model.load_state_dict(torch.load(test_save_path, map_location=device))
        #     with torch.no_grad():
        #         model.eval()
        #         test_iterator = tqdm(test_dataloader, desc=f"Testing Fold {fold}", unit="batch")
        #         for batch_idx, (images, labels) in enumerate(test_iterator):
        #             images, labels = images.to(device), labels.to(device)
        #             outputs_logit = model(images)
        #             loss = loss_fn(outputs_logit, labels)

        #             prob = torch.softmax(outputs_logit, dim=1)
        #             _, predictions = torch.max(prob, dim=1)
        #             test_iterator.set_postfix({
        #                 "loss": f"{loss.item():.4f}"
        #             })
        #             # 更新测试观察器
        #             observer.test_update(loss, prob, predictions, labels)
        #     # 计算测试结果
        #     observer.compute_test_result(len(test_dataloader.dataset))
        #     test_dict[fold] = observer.test_metric
        
        print("\nK-Fold Cross Validation Complete.")
        print("Best Results per Fold:")
        for fold, results in best_train_eval_dict.items():
            print(f"Fold {fold}: {results}")
        
    else:
        # Simple split 80/20
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_data = TransformSubset(train_subset, transform=train_transform)
        val_data = TransformSubset(val_subset, transform=val_transform)
            
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        model = get_model(args.num_classes, args.checkpoint_path, device)
        # model = model.to(device)
        
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = get_scheduler(optimizer, args, train_loader)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        
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
            seed=args.seed
        )
        
        trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, criterion, device, observer)
        trainer.run(args.epochs)
    end = time.time()
    total_seconds = end - start
    # 计算小时、分钟和秒
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    # 打印总训练时间
    print(f"Total training time: {hours}h {minutes}m {seconds}s")