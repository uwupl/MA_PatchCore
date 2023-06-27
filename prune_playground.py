import torch
# from torchvision.models import resnet18
import torch_pruning as tp
import numpy as np
from torchinfo  import summary
from utils.backbone import Backbone, OwnBasicblock, prune_model_nni, prune_output_layer, compress_model_nni
from nni.compression.pytorch.pruning import L2NormPruner as Pruner
from utils.datasets import MVTecDataset
from torch.utils.data import DataLoader
from typing import OrderedDict
from torch import nn
from nni.compression.pytorch.pruning import L1NormPruner
import torch_pruning as tp
from train_main import PatchCore
# channels_not_selected = [12, 19, 11, 22]#np.random.choice(128, 32, replace=False)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r"/mnt/crucial/UNI/IIIT_Muen/MA/MVTechAD") #C:\Users\uwupl\IIIT Muen\MA\productive\mvtec_anomaly_detection
    parser.add_argument('--category', default='own', choices=['carpet', 'bottle', 'cable', 'capsule', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'])
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=224)
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--coreset_sampling_ratio', default=0.01)
    parser.add_argument('--project_root_path', default=r'./test')
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=20)
    args = parser.parse_args()
    return args

# model_non_pruned = Backbone(model_id='RN34', layers_needed=[2], layer_cut=True, prune_output_layer=(False, channels_not_selected), sigmoid_in_last_layer=True)
def get_default_PatchCoreModel():
    '''
    Returns a PatchCore model with default settings.
    '''
    args = get_args()
    model = PatchCore(args=args)
    model.model_id = 'WRN50'
    model.layers_needed = [2,3]
    model.pooling_strategy = 'default' # nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    model.exclude_relu = False # relu won't be used for final layer, in order to not lose negative values
    model.sigmoid_in_last_layer = False # sigmoid will be used for final layer
    model.normalize = False # performs normalization on the feature vector; mean = 0, std = 1
    # backbone reduction
    model.layer_cut = False
    model.prune_output_layer = (False, [])
    # nearest neighbor search
    model.coreset_sampling_ratio = 0.01 #1%
    model.faiss_quantized = False
    model.faiss_standard = False
    model.own_knn = True
    # score calculation
    model.adapted_score_calc = False
    model.n_neighbors = 9
    model.n_next_patches = 5 # only for adapted_score_calc
    # channel reduction
    model.reduce_via_std = False
    model.reduce_via_entropy = False
    model.reduce_via_entropy_normed = False
    model.reduction_factor = 50 # only for reduce_via_std or reduce_via_entropy or reduce_via_entropy_normed
    return model

class QuantizedModel(nn.Module):
    def __init__(self, model):
        super(QuantizedModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def generate_fuse_list(model):
    fuse_list = []
    
    all_names = [name for name, _ in model.named_modules()]

    if '0.0' in all_names:
        print('Own last layer')
        print(all_names)
        fuse_list.append(("0.0", "0.1", "0.2")) # this is always the same
        fuse_list.append(("2.block_1.final_0","2.block_1.final_1", "2.block_1.final_2"))
        fuse_list.append(("2.block_2.final_3","2.block_2.final_4"))#, "2.block_2.final_5"))#, '2.block_2.final_3'))

        fuse_list_2 = [(name, name.replace('conv1', 'bn1'), name.replace('conv1', 'relu')) for name in all_names if name.endswith('conv1')]
        fuse_list_3 = [(name, name.replace('conv2', 'bn2')) for name in all_names if name.endswith('conv2')]

        fuse_list.extend(fuse_list_2)
        fuse_list.extend(fuse_list_3)
    else:
        print('generic resnet')

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
    print(fuse_list)
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
        
from torchvision import transforms
from PIL import Image
from utils.datasets import MVTecDataset

data_transforms = transforms.Compose([
                transforms.Resize((224, 224), Image.ANTIALIAS),
                transforms.ToTensor(),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])]) # from imagenet
gt_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.CenterCrop(224)])

dataset = MVTecDataset(root=r"/mnt/crucial/UNI/IIIT_Muen/MA/MVTechAD/own", transform=data_transforms, gt_transform=gt_transforms, phase='train', half=False)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=12)

# load model
model = Backbone(model_id='RN34', layers_needed=[2], layer_cut=True, prune_output_layer=(False, []), sigmoid_in_last_layer=False, need_for_own_last_layer=False).cpu()
fused_model = model.model.eval()
# fuse model
print(fused_model)
fuse_list = generate_fuse_list(fused_model)
a = fuse_model(fused_model, fuse_list)
# add quantization layers
b = QuantizedModel(a)
# set config for architecture
print('\n\n')
print(b)
print('\n\n')
b.qconfig = torch.quantization.get_default_qconfig('x86') # 'qnnpack'
torch.quantization.prepare(b, inplace=True)
# calibrate using training data
calibrate_model(b, loader, device=torch.device("cpu:0"))
# finally convert into quantized model
c = torch.quantization.convert(b, inplace=True)
c.eval()
print(c)
# test inference
for inputs in loader:
    x, _, _, _, _ = inputs
    y = c(x)
    break
print(y)