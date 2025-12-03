import torch
import torchvision
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import (
    Accuracy, Recall, Precision, Specificity, F1Score,
    ConfusionMatrix, AUROC, MetricCollection, CohenKappa
)
import swanlab
from typing import Literal


class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
            verbose (bool): Print message when early stopping is triggered.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class RuntimeObserver:
    """Observer for monitoring training/validation metrics, logging, and early stopping."""

    def __init__(self, log_dir, device, num_classes=2,
                 task: Literal["binary", "multiclass"] = "multiclass",
                 average: Literal["micro", "macro", "weighted", "none"] = "micro",
                 patience: int = 50, fold: int = 0, **kwargs):
        """
        :param log_dir: Directory to save logs and model weights
        :param device: torch.device
        :param num_classes: Number of output classes
        :param task: 'binary' or 'multiclass'
        :param average: Averaging mode for metrics
        :param patience: Early stopping patience
        :param fold: Current fold number (0 for non-kfold training)
        :param kwargs: Optional metadata (experiment name, random seed, hyperparameters)
        """
        self.log_dir = str(log_dir)
        self.log_file = self.log_dir + 'log.txt'
        self._kwargs = {
            'name': kwargs.get('name', 'None'),
            'seed': kwargs.get('seed', 'None'),
            'hyperparameters': kwargs.get('hyperparameters', {})
        }
        self.device = device
        self.task = task
        self.fold = fold

        # 初始化 best 记录
        self.best_dicts = {
            'epoch': 0, 'confusionMatrix': None, 'Accuracy': 0.,
            'Recall': 0., 'Precision': 0., 'Specificity': 0., 'BalanceAccuracy': 0.,
            'F1': 0., 'AuRoc': 0., 'CohenKappa': 0.
        }

        # ========== Metrics ========== #
        self.train_metric_collection = MetricCollection({
            'confusionMatrix': ConfusionMatrix(num_classes=num_classes, task=task).to(device),
            'Accuracy': Accuracy(num_classes=num_classes, task=task).to(device),
            'Precision': Precision(num_classes=num_classes, task=task, average=average).to(device),
            'Recall': Recall(num_classes=num_classes, task=task, average=average).to(device),
            'Specificity': Specificity(num_classes=num_classes, task=task, average=average).to(device),
            'F1': F1Score(num_classes=num_classes, task=task, average=average).to(device),
            'CohenKappa': CohenKappa(num_classes=num_classes, task=task).to(device)
        }).to(device)

        self.eval_metric_collection = MetricCollection({
            'confusionMatrix': ConfusionMatrix(num_classes=num_classes, task=task).to(device),
            'Accuracy': Accuracy(num_classes=num_classes, task=task).to(device),
            'Precision': Precision(num_classes=num_classes, task=task, average=average).to(device),
            'Recall': Recall(num_classes=num_classes, task=task, average=average).to(device),
            'Specificity': Specificity(num_classes=num_classes, task=task, average=average).to(device),
            'F1': F1Score(num_classes=num_classes, task=task, average=average).to(device),
            'CohenKappa': CohenKappa(num_classes=num_classes, task=task).to(device)
        }).to(device)

        self.test_metric_collection = MetricCollection({
            'confusionMatrix': ConfusionMatrix(num_classes=num_classes, task=task).to(device),
            'Accuracy': Accuracy(num_classes=num_classes, task=task).to(device),
            'Precision': Precision(num_classes=num_classes, task=task, average=average).to(device),
            'Recall': Recall(num_classes=num_classes, task=task, average=average).to(device),
            'Specificity': Specificity(num_classes=num_classes, task=task, average=average).to(device),
            'F1': F1Score(num_classes=num_classes, task=task, average=average).to(device),
            'CohenKappa': CohenKappa(num_classes=num_classes, task=task).to(device)
        }).to(device)

        self.compute_train_auc = AUROC(num_classes=num_classes, task=task).to(device)
        self.compute_eval_auc = AUROC(num_classes=num_classes, task=task).to(device)
        self.compute_test_auc = AUROC(num_classes=num_classes, task=task).to(device)

        # SwanLab会话由外部管理，这里不初始化
        # 用于标识当前fold的前缀
        self.metric_prefix = f"Fold{fold}/" if fold > 0 else ""
        
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        # 记录实验基本信息和超参数
        self.log(f"Experiment: {self._kwargs['name']} | Seed: {self._kwargs['seed']}")
        self.log(f"Hyperparameters: {self._kwargs['hyperparameters']}")
        # 初始化统计
        self.reset()

    # ============================== #
    # ---------- UTILITIES ---------- #
    # ============================== #
    def log(self, info: str):
        print(info)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(info + '\n')

    def reset(self):
        """Reset all epoch-wise statistics."""
        self.total_train_loss = 0.
        self.average_train_loss = 0.
        self.train_metric = {}
        self.train_metric_collection.reset()
        self.compute_train_auc.reset()
        self.train_auc = 0.
        self.train_balance_accuracy = 0.

        self.total_eval_loss = 0.
        self.average_eval_loss = 0.
        self.eval_metric = {}
        self.eval_metric_collection.reset()
        self.compute_eval_auc.reset()
        self.eval_auc = 0.
        self.eval_balance_accuracy = 0.

        self.total_test_loss = 0.
        self.average_test_loss = 0.
        self.test_metric = {}
        self.test_metric_collection.reset()
        self.compute_test_auc.reset()
        self.test_auc = 0.
        self.test_balance_accuracy = 0.


    def train_update(self, loss, prob, prediction, label):
        """Update training metrics."""
        if self.task == 'binary':
            prob_positive = prob[:, 1]
        else:
            prob_positive = prob
        self.total_train_loss += loss.item()
        self.train_metric_collection.update(prediction, label)
        self.compute_train_auc.update(prob_positive, label)

    def eval_update(self, loss, prob, prediction, label):
        """Update evaluation metrics."""
        if self.task == 'binary':
            prob_positive = prob[:, 1]
        else:
            prob_positive = prob
        self.total_eval_loss += loss.item()
        self.eval_metric_collection.update(prediction, label)
        self.compute_eval_auc.update(prob_positive, label)
    
    def test_update(self, loss, prob, prediction, label):
        """Update test metrics."""
        if self.task == 'binary':
            prob_positive = prob[:, 1]
        else:
            prob_positive = prob
        self.total_test_loss += loss.item()
        self.test_metric_collection.update(prediction, label)
        self.compute_test_auc.update(prob_positive, label)


    def compute_result(self, epoch, train_dataset_length, eval_dataset_length):
        """Compute metrics for the epoch and log to SwanLab."""
        self.average_train_loss = self.total_train_loss / train_dataset_length
        self.average_eval_loss = self.total_eval_loss / eval_dataset_length

        self.train_metric = self.train_metric_collection.compute()
        self.eval_metric = self.eval_metric_collection.compute()

        self.train_auc = self.compute_train_auc.compute()
        self.eval_auc = self.compute_eval_auc.compute()

        self.train_balance_accuracy = (self.train_metric['Recall'] + self.train_metric['Specificity']) / 2.0
        self.eval_balance_accuracy = (self.eval_metric['Recall'] + self.eval_metric['Specificity']) / 2.0

        # SwanLab logging with fold prefix
        swanlab.log({
            f'{self.metric_prefix}Loss/train': self.average_train_loss,
            f'{self.metric_prefix}Loss/val': self.average_eval_loss,
            f'{self.metric_prefix}Eval/Accuracy': self.eval_metric['Accuracy'].item(),
            f'{self.metric_prefix}Eval/F1': self.eval_metric['F1'].item(),
            f'{self.metric_prefix}Eval/AUC': self.eval_auc.item(),
            f'{self.metric_prefix}Eval/BalanceAcc': self.eval_balance_accuracy.item(),
            f'{self.metric_prefix}Eval/CohenKappa': self.eval_metric['CohenKappa'].item(),
            f'{self.metric_prefix}Train/Accuracy': self.train_metric['Accuracy'].item(),
            f'{self.metric_prefix}Train/F1': self.train_metric['F1'].item(),
            f'{self.metric_prefix}Train/AUC': self.train_auc.item(),
            f'{self.metric_prefix}Train/BalanceAcc': self.train_balance_accuracy.item(),
        }, step=epoch)

    def print_result(self, e, epochs):
        """Print results for the current epoch."""
        msg_train = (
            f"Epoch [{e}/{epochs}] Train | Loss={self.average_train_loss:.4f}\n"
            f"ConfusionMatrix:\n{self.train_metric['confusionMatrix']}\n"
            f"Acc={self.train_metric['Accuracy']:.4f}, Pre={self.train_metric['Precision']:.4f}, "
            f"Rec={self.train_metric['Recall']:.4f}, Spe={self.train_metric['Specificity']:.4f}, "
            f"F1={self.train_metric['F1']:.4f}, Kappa={self.train_metric['CohenKappa']:.4f}, "
            f"BalAcc={self.train_balance_accuracy:.4f}, AUC={self.train_auc:.4f}\n"
        )
        msg_eval = (
            f"Epoch [{e}/{epochs}] Eval | Loss={self.average_eval_loss:.4f}\n"
            f"ConfusionMatrix:\n{self.eval_metric['confusionMatrix']}\n"
            f"Acc={self.eval_metric['Accuracy']:.4f}, Pre={self.eval_metric['Precision']:.4f}, "
            f"Rec={self.eval_metric['Recall']:.4f}, Spe={self.eval_metric['Specificity']:.4f}, "
            f"F1={self.eval_metric['F1']:.4f}, Kappa={self.eval_metric['CohenKappa']:.4f}, "
            f"BalAcc={self.eval_balance_accuracy:.4f}, AUC={self.eval_auc:.4f}\n"
        )
        self.log(msg_train)
        self.log(msg_eval)

    def get_best(self, epoch):
        """Record best metrics."""
        self.best_dicts['epoch'] = epoch
        self.best_dicts['confusionMatrix'] = self.eval_metric['confusionMatrix'].cpu().numpy()
        self.best_dicts['Accuracy'] = self.eval_metric['Accuracy']
        self.best_dicts['Recall'] = self.eval_metric['Recall']
        self.best_dicts['Precision'] = self.eval_metric['Precision']
        self.best_dicts['Specificity'] = self.eval_metric['Specificity']
        self.best_dicts['BalanceAccuracy'] = self.eval_balance_accuracy
        self.best_dicts['F1'] = self.eval_metric['F1']
        self.best_dicts['CohenKappa'] = self.eval_metric['CohenKappa']
        self.best_dicts['AuRoc'] = self.eval_auc

    def execute(self, e, epochs, train_len, eval_len, fold, model=None):
        """Compute metrics, print, check early stopping, and save best model."""
        self.compute_result(e, train_len, eval_len)
        self.print_result(e, epochs)

        # 更新最佳模型
        if self.eval_balance_accuracy > self.best_dicts['BalanceAccuracy']:
            self.get_best(e)
            if model is not None:
                model_save_path = self.log_dir + f'{self._kwargs["name"]}_best_model_fold{fold}.pth'
                torch.save(model.state_dict(), model_save_path)
                self.log(f"✅ Best model saved to {model_save_path}\n")

        # Early stopping
        self.early_stopping(self.eval_metric['Accuracy'])
        return self.early_stopping.early_stop
    
    def compute_test_result(self, test_dataset_length):
        """Compute test metrics."""
        self.average_test_loss = self.total_test_loss / test_dataset_length

        self.test_metric = self.test_metric_collection.compute()
        self.test_auc = self.compute_test_auc.compute()
        self.test_balance_accuracy = (self.test_metric['Recall'] + self.test_metric['Specificity']) / 2.0
        self.test_metric['AuRoc'] = self.test_auc
        self.test_metric['BalanceAccuracy'] = self.test_balance_accuracy
        
        msg_test = (
            f"Test Results | Loss={self.average_test_loss:.4f}\n"
            f"ConfusionMatrix:\n{self.test_metric['confusionMatrix']}\n"
            f"Acc={self.test_metric['Accuracy']:.4f}, Pre={self.test_metric['Precision']:.4f}, "
            f"Rec={self.test_metric['Recall']:.4f}, Spe={self.test_metric['Specificity']:.4f}, "
            f"F1={self.test_metric['F1']:.4f}, Kappa={self.test_metric['CohenKappa']:.4f}, "
            f"BalAcc={self.test_balance_accuracy:.4f}, AUC={self.test_auc:.4f}\n"
        )
        self.log(msg_test)
        # 保存测试集的混淆矩阵图像（归一化视图，带注释）
        cm = self.test_metric.get('confusionMatrix', None)
        if cm is not None:
            try:
                # 使用更加友好的可视化（按需归一化并显示标签）
                self._save_confusion_matrix(cm, fold='test', normalize=True)
            except Exception as e:
                self.log(f"Failed to save test confusion matrix: {e}")
    

    def finish(self, fold):
        """Log best result summary."""
        best = self.best_dicts
        msg = (
            f"==== Fold {fold} Best Epoch {best['epoch']} ====\n"
            f"Acc={best['Accuracy']:.4f}, Pre={best['Precision']:.4f}, Rec={best['Recall']:.4f}, "
            f"Spe={best['Specificity']:.4f}, Kappa={best['CohenKappa']:.4f}\n"
            f"BalAcc={best['BalanceAccuracy']:.4f}, F1={best['F1']:.4f}, AUC={best['AuRoc']:.4f}\n"
            f"ConfusionMatrix:\n{best['confusionMatrix']}\n"
        )
        self.log(msg)
        
        # 记录最佳结果到swanlab（使用summary记录最终结果）
        swanlab.log({
            f'{self.metric_prefix}Best/Accuracy': float(best['Accuracy']),
            f'{self.metric_prefix}Best/F1': float(best['F1']),
            f'{self.metric_prefix}Best/AUC': float(best['AuRoc']),
            f'{self.metric_prefix}Best/BalanceAcc': float(best['BalanceAccuracy']),
            f'{self.metric_prefix}Best/Epoch': int(best['epoch']),
        })
        
        # 保存混淆矩阵为图片（每折）
        cm = best.get('confusionMatrix', None)
        if cm is not None:
            try:
                self._save_confusion_matrix(cm, fold, normalize=True)
            except Exception as e:
                self.log(f"Failed to save confusion matrix for fold {fold}: {e}")
    def _save_confusion_matrix(self, cm, fold, class_names=None, normalize=False, figsize=(10, 8), cmap='Blues'):
        """Save confusion matrix heatmap to disk with options.

        Args:
            cm: confusion matrix (torch.Tensor or numpy array)
            fold: fold identifier (int or str)
            class_names: list of class label names (optional)
            normalize: whether to normalize rows (recall-normalized)
            figsize: figure size
            cmap: colormap
        """
        # 转为 numpy
        if isinstance(cm, torch.Tensor):
            cm = cm.detach().cpu().numpy()
        cm = np.array(cm)

        # 如果是多维（例如 torchmetrics 可能返回 NxN），确保是二维
        if cm.ndim != 2:
            cm = cm.reshape((cm.shape[0], -1))

        num_classes = cm.shape[0]

        # 计算归一化矩阵（按真实标签行归一化，表示召回率分布）
        cm_display = cm.astype(float)
        if normalize:
            with np.errstate(all='ignore'):
                row_sums = cm_display.sum(axis=1, keepdims=True)
                cm_norm = np.divide(cm_display, row_sums, where=(row_sums != 0))
            show_matrix = cm_norm
        else:
            show_matrix = cm_display

        # 生成默认类名
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]

        # 控制注释显示：类别数量过多时仅显示色块
        annotate = num_classes <= 30

        plt.figure(figsize=figsize)
        sns.set(font_scale=0.9)
        ax = plt.gca()
        im = ax.imshow(show_matrix, interpolation='nearest', cmap=cmap)
        cbar = ax.figure.colorbar(im, ax=ax)

        # 设置刻度与标签
        tick_marks = np.arange(num_classes)
        if num_classes <= 50:
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names, rotation=90, fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)
        else:
            # 类别非常多时隐藏标签以免拥挤
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.grid(False) # 关闭网格线

        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
        ax.set_title(f'Confusion Matrix - {self._kwargs.get("name","exp")} - Fold {fold}')

        # 添加注释：显示绝对值与/或百分比
        if annotate:
            fmt = '.2f' if normalize else 'd'
            thresh = show_matrix.max() / 2.
            for i in range(num_classes):
                for j in range(num_classes):
                    value = show_matrix[i, j]
                    if normalize:
                        text = f"{value:.2%}\n({int(cm[i,j])})"
                    else:
                        text = f"{int(cm[i, j])}"
                    ax.text(j, i, text,
                            ha="center", va="center",
                            color="white" if value > thresh else "black", fontsize=7)

        plt.tight_layout()
        save_path = Path(self.log_dir) / f"{self._kwargs.get('name','exp')}_confusion_fold{fold}.png"
        plt.savefig(str(save_path), dpi=300)
        plt.close()
        self.log(f"Confusion matrix saved to {save_path}")
