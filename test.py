import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs.config import get_config
from data.dataset import OxfordFlowersDataset, get_transforms
from models.get_model import get_model
from utils.observer import RuntimeObserver
from utils.reproducibility import set_global_seed
import torch.nn as nn
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default="/data3/wangchangmiao/shenxy/PublicDataset/temp/jpg",
                        help='Path to dataset')
    parser.add_argument('--test_label_file_path', type=str, 
                        default="/data3/wangchangmiao/shenxy/PublicDataset/temp/test.txt",
                        help='Path to csv file')
    parser.add_argument('--checkpoint', type=str, default="/data3/wangchangmiao/shenxy/Code/flower_classify_3/outputs_20251120_144650/baseline_best_model_fold2.pth")
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

    test_dataset = OxfordFlowersDataset(labels_file=args.test_label_file_path, 
                                        img_dir=args.data_dir, transform=test_transform)
    return test_dataset


if __name__ == '__main__':
    args = parse_args()
    
    # 设置全局随机种子确保可复现性
    set_global_seed(args.seed, deterministic=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = get_model(args.num_classes, args.checkpoint, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    timestamp = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    save_path = f"./{args.save_dir}_{timestamp}/"
    Path(save_path).mkdir(exist_ok=True)
    
    observer = RuntimeObserver(log_dir=save_path, device=device, num_classes=args.num_classes,
                               task='multiclass' if args.num_classes > 2 else 'binary', average='macro',
                               seed=args.seed)
    dataset = build_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
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

