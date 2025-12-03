import torch
import torch.nn as nn
import torchvision
from models.__init__ import *



def get_model(model_name, num_class, pretrained_path, device):
    """获取指定模型架构并加载预训练权重（如有提供）
    Args:
        model_name (str): 模型名称，如 'resnet34'
        num_class (int): 分类类别数
        pretrained_path (str): 预训练权重路径
        device (str): 设备类型 ('cpu' 或 'cuda')
    Returns:
        torch.nn.Module: 构建好的模型
    """

    # model = poolformer_s12(num_classes=1000)
    # model.load_state_dict(torch.load(pretrained_path, weights_only=True))
    # model.head = torch.nn.Linear(model.head.in_features, num_class)
    # self_model = model.to(device)

    if model_name == 'resnet34':
        # model = resnet34(num_classes=1000)
        model = torchvision.models.resnet34(weights=None)

        # If a checkpoint path is provided, try to load state dict safely
        # if pretrained_path:
        # model.load_state_dict(torch.load(pretrained_path, map_location=device))
        model.fc = torch.nn.Linear(model.fc.in_features, num_class)  # 修改全连接层
        model = model.to(device)
    else:
        raise ValueError(f"Model name '{model_name}' is not recognized.")

    # self_model = efficientnetv2_s(num_classes=1000)
    # self_model.load_state_dict(torch.load(pretrained_path, weights_only=True))
    # self_model.head.classifier = torch.nn.Linear(self_model.head.classifier.in_features, num_class)
    # self_model = self_model.to(device)

    # for name, para in model.named_parameters():
    #     # 除head外，其他权重全部冻结
    #     if "head" not in name:
    #         para.requires_grad_(False)
    #     else:
    #         print("training {}".format(name))
    return model

