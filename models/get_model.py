import torch
import torch.nn as nn
import torchvision
from models.__init__ import *



def get_model(num_class, pretrained_path, device):

    # model = poolformer_s12(num_classes=1000)
    # model.load_state_dict(torch.load(pretrained_path, weights_only=True))
    # model.head = torch.nn.Linear(model.head.in_features, num_class)
    # self_model = model.to(device)

    # model = resnet34(num_classes=1000)
    model = torchvision.models.resnet34(weights=None)

    # If a checkpoint path is provided, try to load state dict safely
    # if pretrained_path:
        # model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.fc = torch.nn.Linear(model.fc.in_features, num_class)  # 修改全连接层
    model = model.to(device)



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

