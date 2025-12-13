from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW, SGD, RMSprop, Adadelta
from utils.basic import get_scheduler

experiments = {
    "OxfordFlowers_with_resNet34": {
        # ==============================================================================
        # Model Configuration
        # ==============================================================================
        "model_name": "resnet34",
        "pretrained_path": None,
        "num_classes": 102,  # Number of classes
        "checkpoint_path": None,
        
        # ==============================================================================
        # Dataset Configuration
        # ==============================================================================
        "dataset_name": "OxfordFlowers",
        "data_dir": "/home/shenxiangyuhd/public_dataset/oxfordFlowers/pic",
        "train_eval_label_file_path": "/home/shenxiangyuhd/public_dataset/oxfordFlowers/train_valid.txt",
        "test_label_file_path": "/home/shenxiangyuhd/public_dataset/oxfordFlowers/test.txt",
        "img_size": 224,
        "num_workers": 4,  # Number of data loading workers
        
        # ==============================================================================
        # Training Configuration
        # ==============================================================================
        "trainer_name": "TrainerForOxfordFlowers",
        "loss_fn_name": "CrossEntropyLoss",
        "label_smoothing": 0.1,
        "optimizer_name": "AdamW",
        # "scheduler": get_scheduler,
        
        "batch_size": 64,  # Batch size
        "epochs": 2000,  # Number of epochs
        "patience": 200, # Early stopping patience
        "k_fold": 5,     # K-Fold cross validation (0 or 1 to disable)
        
        # ==============================================================================
        # Hyperparameters
        # ==============================================================================
        "lr": 1e-4, # Learning rate
        "weight_decay": 1e-2, # Weight decay
        "momentum": 0.9,  # SGD Momentum.
        
        # Scheduler settings
        # choices=['lambda', 'step', 'plateau', 'cosine', 'exp', 'onecycle']
        "lr_policy": "onecycle", 
        "lr_decay": 0.95,  # initial lambda decay value
        "niter": 50,  # lr decay step

        # ==============================================================================
        # System & Output Configuration
        # ==============================================================================
        "device": "cuda:4",
        "seed": 455,
        "output_dir": "./result",  # Output directory
        
        # ==============================================================================
        # SwanLab Configuration (Optional Experiment Tracking)
        # ==============================================================================
        "use_swanlab": False,  # Enable SwanLab experiment tracking (default: False for backward compatibility)
        # "swanlab_project": "dl-classification",  # SwanLab project name
        "swanlab_description": "Oxford Flowers Classification with ResNet34",  # Experiment description
        "swanlab_num_samples": 32,  # Number of sample images to log (logged once in K-Fold)
    },
    "CIFAR10_with_resNet34": {
        # ==============================================================================
        # Model Configuration
        # ==============================================================================
        "model_name": "resnet34",
        "pretrained_path": None,
        "num_classes": 10,  # Number of classes
        "checkpoint_path": None,
        
        # ==============================================================================
        # Dataset Configuration
        # ==============================================================================
        "dataset_name": "CIFAR10",
        "data_dir": r"/data3/wangchangmiao/shenxy/PublicDataset/cifar10",
        "train_eval_label_file_path": None,
        "test_label_file_path": None,
        "img_size": 224,
        "num_workers": 4,  # Number of data loading workers
        
        # ==============================================================================
        # Training Configuration
        # ==============================================================================
        "trainer_name": "TrainerForCIFAR10",
        "loss_fn_name": "CrossEntropyLoss",
        "label_smoothing": 0.1,
        "optimizer_name": "AdamW",
        # "scheduler": get_scheduler,
        
        "batch_size": 64,  # Batch size
        "epochs": 2000,  # Number of epochs
        "patience": 200, # Early stopping patience
        "k_fold": 5,     # K-Fold cross validation (0 or 1 to disable)
        
        # ==============================================================================
        # Hyperparameters
        # ==============================================================================
        "lr": 1e-4, # Learning rate
        "weight_decay": 1e-2, # Weight decay
        "momentum": 0.9,  # SGD Momentum.
        
        # Scheduler settings
        # choices=['lambda', 'step', 'plateau', 'cosine', 'exp', 'onecycle']
        "lr_policy": "onecycle", 
        "lr_decay": 0.95,  # initial lambda decay value
        "niter": 50,  # lr decay step

        # ==============================================================================
        # System & Output Configuration
        # ==============================================================================
        "device": "cuda:0",
        "seed": 455,
        "output_dir": "./result",  # Output directory
        
        # ==============================================================================
        # SwanLab Configuration (Optional Experiment Tracking)
        # ==============================================================================
        "use_swanlab": True,  # Enable SwanLab experiment tracking (default: False for backward compatibility)
        # "swanlab_project": "dl-classification",  # SwanLab project name
        "swanlab_description": "CIFAR10 Classification with ResNet34",  # Experiment description
        "swanlab_num_samples": 32,  # Number of sample images to log (logged once in K-Fold)
    },
    "CIFAR100_with_resNet34": {
        # ==============================================================================
        # Model Configuration
        # ==============================================================================
        "model_name": "resnet34",
        "pretrained_path": None,
        "num_classes": 100,  # Number of classes
        "checkpoint_path": None,

        
        # ==============================================================================
        # Dataset Configuration
        # ==============================================================================
        "dataset_name": "CIFAR100",
        "data_dir": r"/data3/wangchangmiao/shenxy/PublicDataset/cifar100",
        "train_eval_label_file_path": None,
        "test_label_file_path": None,
        "img_size": 224,
        "num_workers": 4,  # Number of data loading workers
        
        # ==============================================================================
        # Training Configuration
        # ==============================================================================
        "trainer_name": "TrainerForCIFAR100",
        "loss_fn_name": "CrossEntropyLoss",
        "label_smoothing": 0.1,
        "optimizer_name": "AdamW",
        # "scheduler": get_scheduler,
        
        "batch_size": 64,  # Batch size
        "epochs": 2000,  # Number of epochs
        "patience": 200, # Early stopping patience
        "k_fold": 5,     # K-Fold cross validation (0 or 1 to disable)
        
        # ==============================================================================
        # Hyperparameters
        # ==============================================================================
        "lr": 1e-4, # Learning rate
        "weight_decay": 1e-2, # Weight decay
        "momentum": 0.9,  # SGD Momentum.
        
        # Scheduler settings
        # choices=['lambda', 'step', 'plateau', 'cosine', 'exp', 'onecycle']
        "lr_policy": "onecycle", 
        "lr_decay": 0.95,  # initial lambda decay value
        "niter": 50,  # lr decay step

        # ==============================================================================
        # System & Output Configuration
        # ==============================================================================
        "device": "cuda:0",
        "seed": 455,
        "output_dir": "./result",  # Output directory
        
        # ==============================================================================
        # SwanLab Configuration (Optional Experiment Tracking)
        # ==============================================================================
        "use_swanlab": True,  # Enable SwanLab experiment tracking (default: False for backward compatibility)
        # "swanlab_project": "dl-classification",  # SwanLab project name
        "swanlab_description": "CIFAR100 Classification with ResNet34",  # Experiment description
        "swanlab_num_samples": 32,  # Number of sample images to log (logged once in K-Fold)
    }
}
