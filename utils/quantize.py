import torch
if False:
    from torchy.utils.data import DataLoader
    from torchy import nn
else:
    from torch.utils.data import DataLoader, Dataset
    from torch import nn
from time import perf_counter

import numpy as np
from torchvision import transforms
from PIL import Image

import os
import torchvision
import glob


class QuantizedModel(nn.Module):
    def __init__(self, model, layers_needed=None):
        super(QuantizedModel, self).__init__()
        # self.quant = torch.quantization.QuantStub()
        self.model = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            model,
            torch.ao.quantization.DeQuantStub()
        )
        self.layers_needed = layers_needed
        if self.layers_needed is not None:
            for layer in self.layers_needed:
                self.model[1][layer+3][-1].register_forward_hook(self.hook_q)
        else:
            self.model[-2][-1].register_forward_hook(self.hook_q)
    
    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features
    
    def hook_q(self, module, input, output):
        output = output.dequantize()
        self.features.append(output)

def generate_fuse_list(model):
    fuse_list = []
    
    all_names = [name for name, _ in model.named_modules()]

    if '0.0' in all_names:
        # print('Own last layer')
        # print(all_names)
        fuse_list.append(("0.0", "0.1", "0.2")) # this is always the same
        fuse_list.append(("2.block_1.final_0","2.block_1.final_1", "2.block_1.final_2"))
        fuse_list.append(("2.block_2.final_3","2.block_2.final_4"))#, "2.block_2.final_5"))#, '2.block_2.final_3'))

        fuse_list_2 = [(name, name.replace('conv1', 'bn1'), name.replace('conv1', 'relu')) for name in all_names if name.endswith('conv1')]
        fuse_list_3 = [(name, name.replace('conv2', 'bn2')) for name in all_names if name.endswith('conv2')]

        fuse_list.extend(fuse_list_2)
        fuse_list.extend(fuse_list_3)
    else:
        # print('generic resnet')

        fuse_list.append(("0", "1", "2")) # this is always the same
        
        for name in all_names:
            if name.startswith("4."):
                if name.endswith("conv1"):
                    fuse_list.append((name, name.replace("conv1", "bn1"), name.replace("conv1", "relu")))
                elif name.endswith("conv2"):
                    fuse_list.append((name, name.replace("conv2", "bn2")))
            elif name.startswith("5."):
                if name.endswith("conv1"):
                    fuse_list.append((name, name.replace("conv1", "bn1"), name.replace("conv1", "relu")))
                elif name.endswith("conv2"):
                    fuse_list.append((name, name.replace("conv2", "bn2")))
    
    return fuse_list

def fuse_model(model, fuse_list):
    # print(fuse_list)
    fused_model = torch.quantization.fuse_modules(model, fuse_list, inplace=True)

    return fused_model

def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    # model.eval()
    with torch.inference_mode():
        for inputs in loader:
            # print(inputs[0].shape)
            # inputs = inputs.to(device)
            # # labels = labels.to(device)
            x, _, _, _, _ = inputs
            _ = model(x)
        
def quantize_model_into_qint8(model, layers_needed = None, calibrate = None, category = 'own', cpu_arch = 'x86', dataset_path = r"/mnt/crucial/UNI/IIIT_Muen/MA/MVTechAD/"):
    '''
    Quantizes a model into quint8. Utilizes layer fusion and calibration.
    
    Choices for arg calibrate:
        - None: no calibration in order to load a pretrained model's state dict
        - str(target): calibrate with target domain (train data of target domain)
        - str(mvtec): calibrate with MVTec dataset (stacked training data of all MVTec categories)
        - str(imagenet): calibrate with random subset (size 5000, subject to change) of Imagenet dataset (validation data of Imagenet)
        - str(random): calibrate with random images uniformly distributed in [0, 255]
    '''
    st = perf_counter()    
    print('\nQuantizing model into qint8')
    # load model
    fused_model = model.model.eval()
    
    # fuse model
    fuse_list = generate_fuse_list(fused_model)
    a = fuse_model(fused_model, fuse_list)
    
    # add quantization layers
    b = QuantizedModel(a, layers_needed=layers_needed)
    
    # set config for architecture
    b.qconfig = torch.quantization.get_default_qconfig('fbgemm' if cpu_arch.__contains__('x86') else 'qnnpack')#cpu_arch) # 'qnnpack','x86'
    torch.quantization.prepare(b, inplace=True)
    
    # calibrate using training data
    if calibrate:
        if calibrate.lower().__contains__('target'):
            from .datasets import MVTecDataset, data_transforms, gt_transforms
            data_path = os.path.join(dataset_path, category)
            dataset = MVTecDataset(root=data_path, transform=data_transforms, gt_transform=gt_transforms, phase='train', half=False)
        elif calibrate.lower().__contains__('mvtec'):
            from .datasets import MVTecDataset, data_transforms, gt_transforms
            cats = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'own']
            dataset = None
            for cat in cats:
                if dataset is None:
                    dataset = MVTecDataset(root=os.path.join(dataset_path, cat), transform=data_transforms, gt_transform=gt_transforms, phase='train', half=False)
                else:
                    dataset = torch.utils.data.ConcatDataset([dataset, MVTecDataset(root=os.path.join(dataset_path, cat), transform=data_transforms, gt_transform=gt_transforms, phase='train', half=False)])
        elif calibrate.lower().__contains__('imagenet'):
            from .datasets import Own_Imagenet, data_transforms
            dataset = Own_Imagenet(transform=data_transforms, phase='val')
        elif calibrate.lower().__contains__('random'):
            from .datasets import RandomImageDataset, data_transforms
            dataset = RandomImageDataset(num_images=100, transform=data_transforms)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=12)
        calibrate_model(b, loader, device=torch.device("cpu:0"))
    else:
        print('No calibration performed. It is recommended to load a pretrained model\'s state dict then.')
    
    # finally convert into quantized model
    c = torch.quantization.convert(b, inplace=True)
    c.eval()
    
    # test inference
    if calibrate:
        for inputs in loader:
            x, _, _, _, _ = inputs
            _ = c(x)
            break
    et = perf_counter()
    print(f'Quantization took {(et-st):.2f} seconds')
    
    return c


