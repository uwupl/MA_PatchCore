
import torch
# from torchvision.models import resnet18
import torch_pruning as tp
import numpy as np
from torchinfo  import summary
from backbone import Backbone, OwnBasicblock, prune_model_nni, prune_output_layer, compress_model_nni
from nni.compression.pytorch.pruning import L2NormPruner as Pruner
from datasets import MVTecDataset
from torch.utils.data import DataLoader
from typing import OrderedDict
from torch import nn
from nni.compression.pytorch.pruning import L1NormPruner
import torch_pruning as tp
from train_main import PatchCore
# channels_not_selected = [12, 19, 11, 22]#np.random.choice(128, 32, replace=False)

def quantize_model(model, calibration_data):
    # Collect activation statistics for calibration

    model.eval()
    
    
    
    model.qconfig = torch.quantization.get_default_qconfig('x86')
    # set the qengine to control weight packing
    torch.backends.quantized.engine = 'x86'
    torch.quantization.prepare(model, inplace=True)
    
    # Run calibration data through the model
    with torch.no_grad():
        _ = model(calibration_data)
    
    # Convert the model to a quantized version
    torch.quantization.convert(model, inplace=True)
    
    # quantizer = torch.ao.quantization.QuantStub()
    # model = nn.Sequential(quantizer, model)
    
    return model


calibration_data = torch.randn(8, 3, 224, 224).cpu()
model = Backbone(model_id='RN34', layers_needed=[2], layer_cut=True, prune_output_layer=(False, []), sigmoid_in_last_layer=False).cpu()
# quantizer_for_input = torch.ao.quantization.QuantStub()
# q_input = quantizer_for_input(calibration_data)

# model_quantized = quantize_model(model, calibration_data)

# out = model_quantized(torch.randn(8, 3, 224, 224).cpu())

model_fp32 = nn.Sequential(torch.ao.quantization.QuantStub(), model, torch.ao.quantization.DeQuantStub())
model_fp32.eval()


print(model_fp32)
model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')

# model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv1', 'conv2', 'relu']])

model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)

input_fp32 = torch.randn(8, 3, 224, 224)#.cpu()

model_fp32_prepared(input_fp32)

model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

# run the model, relevant calculations will happen in int8
res = model_int8(input_fp32)

# backend = "x86"
# model.qconfig = torch.quantization.get_default_qconfig(backend)
# torch.backends.quantized.engine = backend
# model_static_quantized = torch.quantization.prepare(model, inplace=False)
# model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)

# model_static_quantized.eval()

# b = model_static_quantized(torch.randn(8, 3, 224, 224).cpu())