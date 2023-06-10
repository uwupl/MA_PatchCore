import torch
import torch.nn.utils.prune as prune
from torchvision import models
import torch.nn as nn
from typing import List, Tuple, OrderedDict
import torch_pruning as tp

class Backbone(nn.Module):
    def __init__(
        self,
        model_id: str,
        layers_needed: List[int],
        layer_cut: bool,
        prune_output_layer: Tuple[bool, List[int]] = (False, []),
        prune_l1_norm: Tuple[bool, float] = (False, 0.0),
        exclude_relu: bool = False,
        sigmoid_in_last_layer: bool = False,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.layers_needed = layers_needed
        self.layer_cut = layer_cut
        self.prune_output_layer = prune_output_layer
        self.prune_l1_norm = prune_l1_norm
        self.exclude_relu = exclude_relu
        self.sigmoid_in_last_layer = sigmoid_in_last_layer
        self.init_features()
        
        if self.model_id.__contains__('WRN50'):
            weights = models.Wide_ResNet50_2_Weights.DEFAULT
            self.model =  models.wide_resnet50_2(weights=weights)
            self.procedure_resnet()    
        elif self.model_id.__contains__('WRN101'):
            weights = models.Wide_ResNet101_2_Weights.DEFAULT
            self.model =  models.wide_resnet101_2(weights=weights)
            self.procedure_resnet()
        elif self.model_id.__contains__('RN18'):
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
            self.procedure_resnet()
        elif self.model_id.__contains__('RN34'):
            weights = models.ResNet34_Weights.DEFAULT
            self.model = models.resnet34(weights=weights)
            self.procedure_resnet()
        elif self.model_id.__contains__('RN50'):
            weights = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights)
            self.procedure_resnet()
            
        elif self.model_id.__contains__('CX_XS'):
            weights = models.ConvNeXt_Tiny_Weights
            self.model = models.convnext_tiny(weights=weights).features
            self.procedure_convnext()
        elif self.model_id.__contains__('CX_S'):
            weights = models.ConvNeXt_Small_Weights
            self.model = models.convnext_small(weights=weights).features
            self.procedure_convnext()
        elif self.model_id.__contains__('CX_M'):
            weights = models.ConvNeXt_Base_Weights
            self.model = models.convnext_base(weights=weights).features
            self.procedure_convnext()
        elif self.model_id.__contains__('CX_L'):
            weights = models.ConvNeXt_Large_Weights
            self.model = models.convnext_large(weights=weights).features
            self.procedure_convnext()
        
        for param in self.model.parameters():
            param.requires_grad = False
        
    def hook_t(self,module, input, output):
        self.features.append(output)
    
    def hook_WideResNet(self, module, input, output):
        '''
        takes the model and prune it if desired. Mainly, hooks are placed here. For Nets with Bottleneck, the hook is placed at the end of the Bottleneck.
        '''
        if output.shape[1] == int(256):
            selected_idx = self.selected_idx_dict[1]
        elif output.shape[1] == int(512):
            selected_idx = self.selected_idx_dict[2]
        elif output.shape[1] == int(1024):
            selected_idx = self.selected_idx_dict[3]
        self.features.append(output[:,selected_idx,:,:])

    def hook_ResNet(self, module, input, output):
        '''
        takes the model and prune it if desired. Mainly, hooks are placed here. For Nets with BasicBlock.
        '''
        if output.shape[1] == int(64):
            selected_idx = self.selected_idx_dict[1]
        elif output.shape[1] == int(128):
            selected_idx = self.selected_idx_dict[2]
        elif output.shape[1] == int(256):
            selected_idx = self.selected_idx_dict[3]
        self.features.append(output[:,selected_idx,:,:])

    def procedure_resnet(self):
        '''
        processes the resnet model. This includes pruning, hooking and cutting the model.
        '''
        if self.layer_cut and not self.prune_output_layer[0]:
            self.model = nn.Sequential(*(list(self.model.children())[0:int(4+max(self.layers_needed))]))
            if self.prune_l1_norm[0]:
                print('prune l1 norm')
                self.model = prune_model_l1_unstrucured(self.model, pruning_perc=self.prune_l1_norm[1])
                print('done')
            if int(1) in self.layers_needed:
                list(self.model.children())[4][-1].register_forward_hook(self.hook_t)
            if int(2) in self.layers_needed:
                list(self.model.children())[5][-1].register_forward_hook(self.hook_t)
            if int(3) in self.layers_needed:
                list(self.model.children())[6][-1].register_forward_hook(self.hook_t)
            if int(4) in self.layers_needed:
                list(self.model.children())[7][-1].register_forward_hook(self.hook_t)

        elif not self.layer_cut and not (self.prune_output_layer[0] or self.exclude_relu):
            if self.prune_l1_norm[0]:
                print('prune l1 norm')
                self.model = prune_model_l1_unstrucured(self.model, pruning_perc=self.prune_l1_norm[1])
                print('done')
            if int(1) in self.layers_needed:
                self.model.layer1[-1].register_forward_hook(self.hook_t)
            if int(2) in self.layers_needed:
                self.model.layer2[-1].register_forward_hook(self.hook_t)
            if int(3) in self.layers_needed:
                self.model.layer3[-1].register_forward_hook(self.hook_t)
            if int(4) in self.layers_needed:
                self.model.layer4[-1].register_forward_hook(self.hook_t)

        elif (self.prune_output_layer[0] or self.exclude_relu) and self.model_id.__contains__('W'):
            layer_to_include = max(self.layers_needed)
            selected_idx_list = self.prune_output_layer[1]
            if len(self.layers_needed) > 1:
                # def hook_adapted(module, input, output)
                self.selected_idx_dict = {layer: [] for layer in self.layers_needed}
                current_boundary = 2**int(7+min(self.layers_needed)) # 1st: 256, 2nd: 512, 3rd: 1024, 4th: 2048
                to_subtract = int(0)
                for k, layer in enumerate(self.layers_needed):
                    if k > 0:
                        to_subtract = current_boundary
                        current_boundary += 2**int(7+layer)
                    self.selected_idx_dict[layer] += [int(channel-to_subtract) for channel in selected_idx_list if channel < current_boundary]
                    selected_idx_list = [channel for channel in selected_idx_list if int(channel-to_subtract) not in self.selected_idx_dict[layer]]
                    
            layers_1 = torch.nn.Sequential(*(list(self.model.children())[:int(4+layer_to_include-1)]))
            layers_2 = torch.nn.Sequential(*(list(self.model.children())[int(4+layer_to_include-1)][:-1]))
            output_layer_tmp = torch.nn.Sequential(*(list(self.model.children())[int(4+layer_to_include-1)][-1:]))
            dict_1 = OrderedDict([(f'final_{i}', module) for i, module in enumerate(output_layer_tmp[-1].children()) if i<2])
            dict_2 = OrderedDict([(f'final_{i}', module) for i, module in enumerate(output_layer_tmp[-1].children()) if i<4 and i>=2])
            dict_3 = OrderedDict([(f'final_{i}', module) for i, module in enumerate(output_layer_tmp[-1].children()) if i<6 and i>=4])
            if layer_to_include == int(1):
                input_size = (1,128,56,56)
            elif layer_to_include == int(2):
                input_size = (1,256,28,28)
            elif layer_to_include == int(3):
                input_size = (1,512,14,14)
            elif layer_to_include == int(4):
                input_size = (1,1024,7,7)
            
            if len(self.layers_needed) > 1:
                output_layer = OwnBottleneck(dict_1, dict_2, dict_3, self.selected_idx_dict[max(self.layers_needed)], input_size, self.exclude_relu, self.sigmoid_in_last_layer)
            else:
                output_layer = OwnBottleneck(dict_1, dict_2, dict_3, self.prune_output_layer[1], input_size, self.exclude_relu, self.sigmoid_in_last_layer)
            del self.model
            self.model = nn.Sequential(layers_1, layers_2, output_layer)

            if self.prune_l1_norm[0]:
                print('prune l1 norm')
                self.model = prune_model_l1_unstrucured(self.model, pruning_perc=self.prune_l1_norm[1])
                print('done')
            
            if len(self.layers_needed) > 1:
                for layer in self.layers_needed[:-1]:
                    list(self.model.children())[0][layer+3][-1].register_forward_hook(self.hook_WideResNet)
            self.model[-1].register_forward_hook(self.hook_t)
        
        elif (self.prune_output_layer[0] or self.exclude_relu) and not self.model_id.__contains__('W'):
            # print('here I am')
            layer_to_include = max(self.layers_needed)
            selected_idx_list = self.prune_output_layer[1]
            if len(self.layers_needed) > 1:
                # def hook_adapted(module, input, output)
                self.selected_idx_dict = {layer: [] for layer in self.layers_needed}
                current_boundary = 2**int(5+min(self.layers_needed)) # for resnet18: 1st: 64, 2nd: 128, 3rd: 256, 4th: 512
                to_subtract = int(0)
                for k, layer in enumerate(self.layers_needed):
                    if k > 0:
                        to_subtract = current_boundary
                        current_boundary += 2**int(5+layer)
                    self.selected_idx_dict[layer] += [int(channel-to_subtract) for channel in selected_idx_list if channel < current_boundary]
                    selected_idx_list = [channel for channel in selected_idx_list if int(channel-to_subtract) not in self.selected_idx_dict[layer]]
            layers_1 = torch.nn.Sequential(*(list(self.model.children())[:int(4+layer_to_include-1)]))
            layers_2 = torch.nn.Sequential(*(list(self.model.children())[int(4+layer_to_include-1)][:-1]))
            output_layer_tmp = torch.nn.Sequential(*(list(self.model.children())[int(4+layer_to_include-1)][-1:]))
            dict_1 = OrderedDict([(f'final_{i}', module) for i, module in enumerate(output_layer_tmp[-1].children()) if i<3])
            dict_2 = OrderedDict([(f'final_{i}', module) for i, module in enumerate(output_layer_tmp[-1].children()) if i>=3])
            if layer_to_include == int(1):
                input_size = (1,64,56,56)
            elif layer_to_include == int(2):
                input_size = (1,128,28,28)
            elif layer_to_include == int(3):
                input_size = (1,256,14,14)
            elif layer_to_include == int(4):
                input_size = (1,512,7,7)
            if len(self.layers_needed) > 1:
                output_layer = OwnBasicblock(dict_1, dict_2, self.selected_idx_dict[max(self.layers_needed)], input_size, self.exclude_relu, self.sigmoid_in_last_layer)
            else:
                output_layer = OwnBasicblock(dict_1, dict_2, self.prune_output_layer[1], input_size, self.exclude_relu, self.sigmoid_in_last_layer)  
            del self.model
            self.model = nn.Sequential(layers_1, layers_2, output_layer)

            if self.prune_l1_norm[0]:
                print('prune l1 norm')
                self.model = prune_model_l1_unstrucured(self.model, pruning_perc=self.prune_l1_norm[1])
                print('done')
            
            if len(self.layers_needed) > 1:
                for layer in self.layers_needed[:-1]:
                    list(self.model.children())[0][layer+3][-1].register_forward_hook(self.hook_ResNet)
            self.model[-1].register_forward_hook(self.hook_t)

    def procedure_convnext(self):
        if self.layer_cut and not self.prune_output_layer[0]:
            self.model = nn.Sequential(*(list(self.model.children())[0:int(2*max(self.layers_needed))]))
            if int(1) in self.layers_needed:
                list(self.model.children())[1][-1].block.register_forward_hook(self.hook_t)
            if int(2) in self.layers_needed:
                list(self.model.children())[3][-1].block.register_forward_hook(self.hook_t)
            if int(3) in self.layers_needed:
                list(self.model.children())[5][-1].block.register_forward_hook(self.hook_t)
            if int(4) in self.layers_needed:
                list(self.model.children())[7][-1].block.register_forward_hook(self.hook_t)
        elif not self.layer_cut and not self.prune_output_layer[0]:
            if int(1) in self.layers_needed:
                self.model[1][-1].block.register_forward_hook(self.hook_t)
            if int(2) in self.layers_needed:
                self.model[3][-1].block.register_forward_hook(self.hook_t)
            if int(3) in self.layers_needed:
                self.model[5][-1].block.register_forward_hook(self.hook_t)
            if int(4) in self.layers_needed:
                self.model[7][-1].block.register_forward_hook(self.hook_t)
    
    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

def prune_model_l1_unstrucured(model, pruning_perc):
    '''
    Prune the model with the given pruning_perc. Decisions are made based on the L1 norm of the weights. 
    '''
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_perc)
            # prune.remove(module, 'weight')

    return model


    # def prune_model_l1_unstrucured(self, pruning_perc):
    #     '''
    #     Prune the model with the given pruning_perc. Decisions are made based on the L1 norm of the weights. 
    #     '''
    #     this_model = self.model#.copy()
    #     del self.model
    #     for name, module in this_model.named_modules():
    #         if isinstance(module, nn.Conv2d):
    #             prune.l1_unstructured(module, name='weight', amount=pruning_perc)
    #             #prune.remove(module, 'weight')
    #     self.model = this_model

class OwnBottleneck(torch.nn.Module):
    def __init__(self, block_1, block_2, block_3, idx_selected, input_size, exclude_relu = False, sigmoid_in_last_layer=False):
        '''
        just pass OderedDicts, bottleneck like layer is created. For ResNet with Bottlenecks. 
        '''
        super().__init__()
        self.block_1 = torch.nn.Sequential(block_1)
        self.block_2 = torch.nn.Sequential(block_2)
        self.block_3 = torch.nn.Sequential(block_3)
        self.relu = torch.nn.ReLU(inplace=True)
        if exclude_relu:
            self.output_activation = torch.nn.Identity(inplace=True)
        elif sigmoid_in_last_layer:
            self.output_activation = torch.nn.Sigmoid(inplace=True)
        else:
            self.output_activation = torch.nn.ReLU(inplace=True)
        self.idx_selected = idx_selected
        if len(self.idx_selected) > 0:
            channels_not_selected = [i for i in range(input_size[1]*2) if i not in self.idx_selected]
            DG = tp.DependencyGraph().build_dependency(self.block_3, example_inputs=torch.rand(input_size))
            group = DG.get_pruning_group(self.block_3.final_4, tp.prune_conv_out_channels, idxs=channels_not_selected)
            print(group)
            if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.  
                group.prune()
        
    def forward(self, x):
        identity = x
        
        out = self.block_1(x)
        out = self.relu(out)
        
        out = self.block_2(out)
        out = self.relu(out)
        
        out = self.block_3(out)
        identity = identity[:,self.idx_selected,...]
        
        out += identity
        out = self.output_activation(out)
                
        return out
    
class OwnBasicblock(torch.nn.Module):
    def __init__(self, block_1, block_2, idx_selected, input_size, exclude_relu = False, sigmoid_in_last_layer=False):
        '''
        just pass OderedDicts, bottleneck like layer is created. For ResNet with Basicblocks. 
        '''
        super().__init__()
        self.block_1 = torch.nn.Sequential(block_1)
        self.block_2 = torch.nn.Sequential(block_2)
        if exclude_relu:
            self.output_activation = torch.nn.Identity(inplace=True)
        elif sigmoid_in_last_layer:
            self.output_activation = torch.nn.Sigmoid(inplace=True)
        else:
            self.output_activation = torch.nn.ReLU(inplace=True)
        # self.output_activation = 
        # self.sigmoid = torch.nn.Sigmoid(inplace=True)
        self.idx_selected = idx_selected
        if len(self.idx_selected) > 0:
            channels_not_selected = [i for i in range(input_size[1]) if i not in self.idx_selected]
            DG = tp.DependencyGraph().build_dependency(self.block_2, example_inputs=torch.rand(input_size))
            group = DG.get_pruning_group(self.block_2.final_3, tp.prune_conv_out_channels, idxs=channels_not_selected)
            print(group)
            if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.  
                group.prune()
        
    def forward(self, x):
        identity = x
        out = self.block_1(x)
        out = self.block_2(out)
        identity = identity[:,self.idx_selected,...]
        out += identity
        out = self.output_activation(out)
        # out = self.relu(out)
        return out
    
    
# class OwnBottleneck_wo_ReLu(torch.nn.Module):
#     def __init__(self, block_1, block_2, block_3, idx_selected, input_size):
#         '''
#         just pass OderedDicts, bottleneck like layer is created. For ResNet with Bottlenecks. 
#         '''
#         super().__init__()
#         self.block_1 = torch.nn.Sequential(block_1)
#         self.block_2 = torch.nn.Sequential(block_2)
#         self.block_3 = torch.nn.Sequential(block_3)
#         # self.relu = torch.nn.ReLU(inplace=True)
#         self.idx_selected = idx_selected
#         channels_not_selected = [i for i in range(input_size[1]*2) if i not in self.idx_selected]
#         DG = tp.DependencyGraph().build_dependency(self.block_3, example_inputs=torch.rand(input_size))
#         group = DG.get_pruning_group(self.block_3.final_4, tp.prune_conv_out_channels, idxs=channels_not_selected)
#         print(group)
#         if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.  
#             group.prune()
        
#     def forward(self, x):
#         identity = x
        
#         out = self.block_1(x)
#         out = self.relu(out)
        
#         out = self.block_2(out)
#         out = self.relu(out)
        
#         out = self.block_3(out)
#         identity = identity[:,self.idx_selected,...]
        
#         out += identity
#         # out = self.relu(out)
                
#         return out
    
# class OwnBasicblock_wo_ReLu(torch.nn.Module):
#     def __init__(self, block_1, block_2, idx_selected, input_size):
#         '''
#         just pass OderedDicts, bottleneck like layer is created. For ResNet with Basicblocks. 
#         '''
#         super().__init__()
#         self.block_1 = torch.nn.Sequential(block_1)
#         self.block_2 = torch.nn.Sequential(block_2)
#         # self.relu = torch.nn.ReLU(inplace=True)
#         self.idx_selected = idx_selected
#         channels_not_selected = [i for i in range(input_size[1]) if i not in self.idx_selected]
#         DG = tp.DependencyGraph().build_dependency(self.block_2, example_inputs=torch.rand(input_size))
#         group = DG.get_pruning_group(self.block_2.final_3, tp.prune_conv_out_channels, idxs=channels_not_selected)
#         print(group)
#         if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.  
#             group.prune()
        
#     def forward(self, x):
#         identity = x
#         out = self.block_1(x)
#         out = self.block_2(out)
#         identity = identity[:,self.idx_selected,...]
#         out += identity
#         # out = self.relu(out)
#         return out