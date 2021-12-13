from os import PathLike
from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.functional import Tensor
from torch.optim import lr_scheduler
import torchvision.transforms.functional as F


DEVICE = 'cuda'
WIDTH = 1200
HEIGHT = 800

def show(imgs: Union[Tensor, List[Tensor]], figsize = (8, 8)):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize = figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img.type(torch.uint8))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    lr_scheduler: lr_scheduler.StepLR, epoch: int, loss: float,
                    path: PathLike):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': loss,
            }, path)


def load_checkpoint(path: PathLike, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: lr_scheduler.StepLR):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, scheduler, epoch, loss



def export_to_onnx(model: torch.nn.Module, path: PathLike):
    cpu_device = torch.device('cpu') 
    x = torch.randn((1, 3, HEIGHT, WIDTH), device = cpu_device)
    model.to(cpu_device)

    # finally convert pytorch model to onnx 
    torch.onnx.export(
        model, x , 
        path, verbose=True, 
        export_params=True, 
        do_constant_folding=True, 
        opset_version=11,
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes = {'input' : {0 : 'batch_size'},    # variable length axes
                    'output' : {0 : 'batch_size'}}
    )
    