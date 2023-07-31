import torch
from utils.quantize import quantize_model_into_qint8
from utils.backbone import Backbone

cpu_arch = 'qnnpack'
# dataset_path = has to be set in dataset.py

for model_id in ['RN18', 'RN34']:
    for layers_needed in [[2],[3],[2,3],[1,2,3]]:
        model_fp32 = Backbone(model_id=model_id, layers_needed=layers_needed, layer_cut=True)
        model_qint8 = quantize_model_into_qint8(model_fp32, layers_needed=layers_needed, calibrate='imagenet',
                                                cpu_arch=cpu_arch, dataset_path=dataset_path)
        torch.save(model_qint8.state_dict(), f'./models/{model_id}_layers_{layers_needed}_qint8_{cpu_arch}.pth')