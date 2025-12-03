import numpy as np
from torch.nn import init
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.optim import lr_scheduler

def get_optimizer(optimizer_name, parameters, **kwargs):
    """
    获取优化器实例

    Args:
        parameters: 模型参数
        lr: 学习率
        weight_decay: 权重衰减
        optimizer_name: 优化器名称（如 'Adam', 'SGD', 'RMSprop', 'Adadelta'）

    Returns:
        optimizer: 对应的优化器实例
    """
    lr = kwargs.get('lr', 1e-3)
    weight_decay = kwargs.get('weight_decay', 0)

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = kwargs.get('momentum', 0.9)
        optimizer = optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adadelta':
        optimizer = optim.Adadelta(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported. Please implement it in get_optimizer function.")

    return optimizer

def get_scheduler(optimizer, opt, train_loader=None):
    """
    定义学习率调度器（scheduler）

    Args:
        optimizer: torch.optim 优化器
        opt: 参数配置对象，需包含 lr_policy 等属性
        train_loader: 仅在 OneCycleLR 时需要，用于计算 steps_per_epoch

    Returns:
        scheduler: 对应的学习率调度器
    """

    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)

    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            threshold=0.01,
            patience=5
        )

    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=3,
            eta_min=0.00001
        )

    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay)

    elif opt.lr_policy == 'onecycle':
        if train_loader is None:
            raise ValueError("❌ OneCycleLR 策略需要传入 train_loader 参数以计算 steps_per_epoch")
        if not hasattr(opt, "epochs"):
            raise ValueError("❌ OneCycleLR 策略需要 opt.epochs 参数")

        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=opt.lr * 5,  # 峰值学习率（通常为初始lr的3~10倍）
            steps_per_epoch=len(train_loader),
            epochs=opt.epochs,
            anneal_strategy='cos',  # 余弦退火
            pct_start=0.1,          # 10% 的时间用于 warm-up
            # div_factor=25.0,        # 初始学习率 = max_lr / div_factor
            # final_div_factor=1e4,   # 最低学习率 = max_lr / final_div_factor
            # three_phase=False       # 可选：是否三阶段策略
        )

    else:
        raise NotImplementedError(f'learning rate policy [{opt.lr_policy}] is not implemented')

    return scheduler

# 初始化权重
def init_weights(net, init_type='normal', gain=0.02):
    """
    initialize the network weights
    :param net: the network
    :param init_type:  initialized method
    :param gain: corresponding gain
    :return: the initialized network
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=None):
    """
    initial the network
    :param net:  to be initialized network
    :param init_type:  initialized method
    :param gain: corresponding gain
    :param gpu_ids: the gpu ids
    :return: the initialized network
    """
    # if gpu_ids is None:
    #     gpu_ids = [-1, ]
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net)
        net.cuda()
    init_weights(net, init_type, gain=init_gain)
    return net

