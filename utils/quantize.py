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
from .datasets import MVTecDataset
import os
import torchvision


class QuantizedModel(nn.Module):
    def __init__(self, model, layers_needed=None):
        super(QuantizedModel, self).__init__()
        # self.quant = torch.quantization.QuantStub()
        self.model = nn.Sequential(
            torch.quantization.QuantStub(),
            model,
            torch.quantization.DeQuantStub()
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
    fused_model = torch.quantization.fuse_modules(model, fuse_list)

    return fused_model

def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    model.eval()

    for inputs in loader:
        # print(inputs[0].shape)
        # inputs = inputs.to(device)
        # # labels = labels.to(device)
        x, _, _, _, _ = inputs
        _ = model(x)
        
def quantize_model_into_qint8(model, layers_needed = None, category = 'own', cpu_arch = 'x86', dataset_path = r"/mnt/crucial/UNI/IIIT_Muen/MA/MVTechAD/"):
    '''
    Quantizes a model into quint8. Utilizes layer fusion and calibration.
    '''
    st = perf_counter()    
    data_transforms = transforms.Compose([
                    transforms.Resize((256, 256), Image.ANTIALIAS),
                    transforms.ToTensor(),
                    transforms.CenterCrop(224),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]) # from imagenet
    gt_transforms = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.CenterCrop(224)])
    
    
    
    # cats = ['bottle','own', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    # if False:
    # first option: calibrate with target domain
    # data_path = os.path.join(dataset_path, category)
    # dataset = MVTecDataset(root=data_path, transform=data_transforms, gt_transform=gt_transforms, phase='train', half=False)
    
    # second option: calibrate with random images
    # dataset = RandomImageDataset(num_images=1000, transform=data_transforms)
    
    # third option: calibrate with imagenet TODO
    # dataset = torchvision.datasets.ImageNet(root='/mnt/crucial/UNI/IIIT_Muen/MA/ILSVRC2012/', split='val', transform=data_transforms)
    
    # fourth option: calibrate with stacked MCTec datasets
    cats = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'own']
    dataset = None
    for cat in cats:
        if dataset is None:
            dataset = MVTecDataset(root=os.path.join(dataset_path, cat), transform=data_transforms, gt_transform=gt_transforms, phase='train', half=False)
        else:
            dataset = torch.utils.data.ConcatDataset([dataset, MVTecDataset(root=os.path.join(dataset_path, cat), transform=data_transforms, gt_transform=gt_transforms, phase='train', half=False)])
        
            
    loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=12)

    # load model
    # model = Backbone(model_id='RN34', layers_needed=[2], layer_cut=True, prune_output_layer=(False, []), sigmoid_in_last_layer=False, need_for_own_last_layer=False, quantize_qint8=True).cpu()
    fused_model = model.model.eval()
    # fuse model
    # print(fused_model)
    fuse_list = generate_fuse_list(fused_model)
    a = fuse_model(fused_model, fuse_list)
    # add quantization layers
    b = QuantizedModel(a, layers_needed=layers_needed)
    
    # set config for architecture
    print('\n\n')
    print(b)
    print('\n\n')
    b.qconfig = torch.quantization.get_default_qconfig('fbgemm')#cpu_arch) # 'qnnpack','x86'
    torch.quantization.prepare(b, inplace=True)
    # calibrate using training data
    calibrate_model(b, loader, device=torch.device("cpu:0"))
    # finally convert into quantized model
    c = torch.quantization.convert(b, inplace=True)
    c.eval()
    # print(c)
    # test inference
    for inputs in loader:
        x, _, _, _, _ = inputs
        # print(x.shape)
        y = c(x)
        break
    # print(y)
    et = perf_counter()
    print(f'Quantization took {(et-st):.2f} seconds')
    
    return c


class RandomImageDataset(Dataset):
    def __init__(self, num_images, transform=None):
        self.num_images = num_images
        self.transform = transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Generate a random image
        image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)

        # Convert numpy array to PIL image
        image = transforms.ToPILImage()(image)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, 0, 0, 0, 0