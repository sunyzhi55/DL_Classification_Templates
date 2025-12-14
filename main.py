import time
from pathlib import Path
import copy
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
from configs.config import get_config
from data.dataset import get_transforms, get_dataset
from engine.trainer import get_trainer
from models.get_model import get_model
from utils.basic import get_optimizer, get_scheduler
from utils.loss_function import get_loss_function
from utils.observer import RuntimeObserver
from utils.reproducibility import set_global_seed, make_reproducible_split


class TransformSubset(Dataset):
    """Apply transforms lazily to items from a subset."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        item = self.subset[index]
        if isinstance(item, dict):
            image, label = item['image'], item['label']
            if self.transform:
                image = self.transform(image)
            return {'image': image, 'label': label, **{k: v for k, v in item.items() if k not in ['image', 'label']}}
        else:
            image, label = item
            if self.transform:
                image = self.transform(image)
            return {'image': image, 'label': label}

    def __len__(self):
        return len(self.subset)


def build_dataloader(dataset, args, shuffle):
    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        'num_workers': args.num_workers,
        'pin_memory': torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs.update({'prefetch_factor': 2, 'persistent_workers': True})
    return DataLoader(dataset, **loader_kwargs)


def build_components(args, device, train_loader):
    model = get_model(args.model_name, args.num_classes, args.checkpoint_path, device)
    optimizer = get_optimizer(
        args.optimizer_name,
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        # momentum=getattr(args, 'momentum', 0.9)
    )
    scheduler = get_scheduler(optimizer, args, train_loader)
    criterion = get_loss_function(args.loss_fn_name, device, label_smoothing=args.label_smoothing)
    return model, optimizer, scheduler, criterion


def create_observer(args, device, log_root):
    return RuntimeObserver(
        log_dir=log_root,
        device=device,
        num_classes=args.num_classes,
        task="multiclass" if args.num_classes > 2 else "binary",
        average='macro' if args.num_classes > 2 else 'micro',
        patience=args.patience,
        # name=args.exp_name,
        # seed=args.seed,
        hyperparameters=vars(args)
    )


def train_single_split(args, device, full_dataset, train_transform, val_transform):
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_subset, val_subset = make_reproducible_split(full_dataset, [train_size, val_size], seed=args.seed)

    train_loader = build_dataloader(TransformSubset(train_subset, transform=train_transform), args, shuffle=True)
    val_loader = build_dataloader(TransformSubset(val_subset, transform=val_transform), args, shuffle=False)

    model, optimizer, scheduler, criterion = build_components(args, device, train_loader)
    observer = create_observer(args, device, args.save_dir)

    if observer.swanlab_logger.enabled:
        observer.swanlab_logger.log_sample_images(full_dataset, fold=0)

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

    if observer.swanlab_logger.enabled:
        observer.swanlab_logger.finish()


def train_k_fold(args, device, full_dataset, train_transform, val_transform):
    print(f"Starting {args.k_fold}-Fold Cross Validation")
    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    best_train_eval_dict = {}
    last_observer = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"\n============================")
        print(f"🔥 开始训练 Fold {fold + 1} / {args.k_fold}")
        print("============================\n")

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = build_dataloader(TransformSubset(train_subset, transform=train_transform), args, shuffle=True)
        val_loader = build_dataloader(TransformSubset(val_subset, transform=val_transform), args, shuffle=False)

        model, optimizer, scheduler, criterion = build_components(args, device, train_loader)
        observer = create_observer(args, device, args.save_dir)
        last_observer = observer

        if fold == 0 and observer.swanlab_logger.enabled:
            observer.swanlab_logger.log_sample_images(full_dataset, fold=fold)
        elif observer.swanlab_logger.enabled:
            observer.swanlab_logger._images_logged = True  # prevent duplicate image logs

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
            fold=fold + 1
        )
        trainer.run(args.epochs)
        best_train_eval_dict[fold + 1] = copy.deepcopy(observer.best_dicts)

    print("\nK-Fold Cross Validation Complete.")
    print("Best Results per Fold:")
    for fold, results in best_train_eval_dict.items():
        print(f"Fold {fold}: {results}")

    if last_observer and last_observer.swanlab_logger.enabled:
        last_observer.swanlab_logger.finish()


def main():
    args = get_config()
    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transforms = get_transforms(args.img_size)
    train_transform = transforms['train_transforms']
    val_transform = transforms['validation_transforms']

    full_dataset = get_dataset(
        args.dataset_name,
        img_dir=args.data_dir,
        labels_file=args.train_eval_label_file_path,
        transform=None
    )
    print(f"Total images in full dataset: {len(full_dataset)}")

    start_time = time.time()
    if args.k_fold and args.k_fold > 1:
        train_k_fold(args, device, full_dataset, train_transform, val_transform)
    else:
        train_single_split(args, device, full_dataset, train_transform, val_transform)
    
    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"Total training time: {hours}h {minutes}m {seconds}s")


if __name__ == '__main__':
    main()