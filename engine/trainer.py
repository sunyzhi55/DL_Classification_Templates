import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from pathlib import Path

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, device, observer, fold=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.observer = observer
        self.fold = fold
        
        self.start_epoch = 1

    def train_one_epoch(self, epoch):
        self.model.train()
        self.observer.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for ii, batch in enumerate(pbar):
            image = batch.get('image')
            label = batch.get('label')
            image, label = image.to(self.device), label.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(image)
            loss = self.criterion(outputs, label)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 训练循环中
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()  # OneCycleLR在每个batch更新学习率
            
            
            prob = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(prob, dim=1)

            self.observer.train_update(loss, prob, predictions, label)

            current_lr = self.optimizer.param_groups[0]['lr']
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "LR": f"{current_lr:.6f}"
            })

    def validate(self, epoch):
        self.model.eval()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for ii, batch in enumerate(pbar):
                image = batch.get('image')
                label = batch.get('label')
                image, label = image.to(self.device), label.to(self.device)

                outputs = self.model(image)
                loss = self.criterion(outputs, label)
                
                prob = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(prob, dim=1)
                
                self.observer.eval_update(loss, prob, predictions, label)

    def run(self, epochs):
        self.observer.log(f'Fold {self.fold}')
        for epoch in range(self.start_epoch, epochs + 1):
            self.train_one_epoch(epoch)
            self.validate(epoch)
            
            if self.observer.execute(epoch, epochs, len(self.train_loader.dataset), len(self.val_loader.dataset), self.fold, self.model):
                print("Early stopping triggered.")
                break
            
            # epoch结束后
            if self.scheduler is not None:
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()  # epoch级别更新
        self.observer.finish(self.fold)

def get_trainer(trainer_name, **kwargs):
    
    return Trainer(**kwargs)
    