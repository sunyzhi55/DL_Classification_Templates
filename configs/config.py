import argparse
from pathlib import Path
from datetime import datetime
from configs.experiments_object import experiments

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Deep Learning Classification Project')
        
        # Experiment
        self.parser.add_argument('--exp_name', type=str, default='CIFAR10_with_resNet34', help='Experiment name')
        # self.parser.add_argument('--seed', type=int, default=455, help='Random seed')
        # self.parser.add_argument('--device', type=str, default='cuda:4', help='Device: cuda or cpu')
        # self.parser.add_argument('--output_dir', type=str, default='./result', help='Output directory')


        
        # Data
        # self.parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root')
        # self.parser.add_argument('--train_label_file_path', type=str, default=None, help='Path to train label file (txt/csv)')
        # self.parser.add_argument('--val_label_file_path', type=str, default=None, help='Path to val label file (txt/csv)')

        # self.parser.add_argument('--data_dir', type=str, default=r"/data3/wangchangmiao/shenxy/PublicDataset/temp/jpg", help='Path to dataset')
        # self.parser.add_argument('--train_eval_label_file_path', type=str, default="/data3/wangchangmiao/shenxy/PublicDataset/temp/train_valid.txt", help='Path to csv file')
        # self.parser.add_argument('--test_label_file_path', type=str, default="/data3/wangchangmiao/shenxy/PublicDataset/temp/test.txt", help='Path to csv file')

        # self.parser.add_argument('--img_size', type=int, default=224, help='Image size')
        # self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        # self.parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
        # self.parser.add_argument('--num_classes', type=int, default=102, help='Number of classes')
        
        # # Model
        # self.parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to load checkpoint')
        
        # # Training
        # self.parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
        # self.parser.add_argument('--patience', type=int, default=200, help='Early stopping patience')
        # self.parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
        # self.parser.add_argument('--weight_decay', type=float, default=0.002, help='Weight decay')
        # self.parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")

        # self.parser.add_argument('--lr_policy', type=str, default='onecycle',
        #                 choices=['lambda', 'step', 'plateau', 'cosine', 'exp', 'onecycle'], help='Scheduler to use')
        # self.parser.add_argument('--lr_decay', type=float, default=0.95, help='initial lambda decay value')
        # self.parser.add_argument('--niter', type=int, default=50, help='lr decay step')
        
        # # K-Fold
        # self.parser.add_argument('--k_fold', type=int, default=5, help='K-Fold cross validation (0 to disable)')
        
    def parse(self):
        args = self.parser.parse_args()
        
        # Load experiment config from dictionary
        if args.exp_name in experiments:
            exp_config = experiments[args.exp_name]
            print(f"Loading configuration for experiment: {args.exp_name}")
            for key, value in exp_config.items():
                # Overwrite args with values from the dictionary
                setattr(args, key, value)
        else:
            raise ValueError(f"Experiment '{args.exp_name}' not found in experiments dictionary. Please check configs/experiments_object.py")
        
        # Ensure output_dir exists in args, default to './outputs' if not provided in experiment config
        if not hasattr(args, 'output_dir'):
            args.output_dir = './outputs'

        # Create output directory with timestamp (use pathlib.Path)
        timestamp = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
        save_path = f"./{args.output_dir}_{timestamp}_{args.exp_name}/"
        Path(save_path).mkdir(exist_ok=True, parents=True)
        args.save_dir = str(save_path)
        
        return args

def get_config():
    config = Config()
    return config.parse()
