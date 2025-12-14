import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import OxfordFlowersDataset, get_transforms, get_dataset
from models.get_model import get_model
from utils.observer import RuntimeObserver
from utils.reproducibility import set_global_seed
import torch.nn as nn
from datetime import datetime
import torchvision


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default="/data3/wangchangmiao/shenxy/PublicDataset/oxfordFlowers/jpg",
                        help='Path to dataset')
    parser.add_argument('--test_label_file_path', type=str, 
                        default="/data3/wangchangmiao/shenxy/PublicDataset/oxfordFlowers/test.txt",
                        help='Path to csv file')
    parser.add_argument('--checkpoint', type=str, default="/data3/wangchangmiao/shenxy/Code/DL_Classification_Templates/outputs_20251122_210542_oxfordFlowers/baseline_best_model_fold4.pth")
    parser.add_argument('--model_name', type=str, default='resnet34')
    parser.add_argument('--exp_name', type=str, default='oxfordFlowers_with_resNet34')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_classes', type=int, default=102)
    parser.add_argument('--save_dir', type=str, default='./test_outputs')
    parser.add_argument('--seed', type=int, default=555)
    return parser.parse_args()


def build_dataset(args):
    test_transform = get_transforms(args.img_size)['test_transforms']

    # test_dataset = OxfordFlowersDataset(labels_file=args.test_label_file_path, img_dir=args.data_dir, transform=test_transform)
    test_dataset= torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=test_transform)

    return test_dataset


if __name__ == '__main__':
    args = parse_args()
    
    # 设置全局随机种子确保可复现性
    set_global_seed(args.seed, deterministic=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = get_model(args.model_name, args.num_classes, args.checkpoint, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    timestamp = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    save_path = Path(f"{args.save_dir}_{timestamp}/")
    save_path.mkdir(exist_ok=True, parents=True)
    
    observer = RuntimeObserver(log_dir=str(save_path), device=device, num_classes=args.num_classes,
                               task='multiclass' if args.num_classes > 2 else 'binary', average='macro',
                               hyperparameters={'exp_name': args.exp_name, 'seed': args.seed})
    dataset = build_dataset(args)
    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': args.num_workers,
        'pin_memory': torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs.update({'prefetch_factor': 2, 'persistent_workers': True})
    dataloader = DataLoader(dataset, **loader_kwargs)
    pbar = tqdm(dataloader, desc="Testing")

    with torch.no_grad():
        for ii, batch in enumerate(pbar):
            images = batch.get("image").to(device)
            labels = batch.get("label").to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            prob = torch.softmax(outputs, dim=1)
            _, preds = torch.max(prob, dim=1)
            observer.test_update(loss, prob, preds, labels)

    observer.compute_test_result(len(dataloader.dataset))
    print('Test finished. Metrics:')
    for k, v in observer.test_metric.items():
        print(f"{k}: {v}")

