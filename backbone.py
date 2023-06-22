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
        prune_torch_pruning: Tuple[bool, List[int]] = (False, 0.0),
        prune_l1_norm: Tuple[bool, float] = (False, 0.0),
        exclude_relu: bool = False,
        sigmoid_in_last_layer: bool = False,
        need_for_own_last_layer: bool = False,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.layers_needed = layers_needed
        self.layer_cut = layer_cut
        self.prune_output_layer = prune_output_layer
        self.prune_l1_unstructured = prune_l1_norm#
        self.prune_torch_pruning = prune_torch_pruning
        self.exclude_relu = exclude_relu
        self.sigmoid_in_last_layer = sigmoid_in_last_layer
        self.need_for_own_last_layer = sigmoid_in_last_layer or exclude_relu or prune_output_layer[0] or need_for_own_last_layer
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
        if self.layer_cut and not self.need_for_own_last_layer:
            self.model = nn.Sequential(*(list(self.model.children())[0:int(4+max(self.layers_needed))]))
            if self.prune_l1_unstructured[0]:
                print('prune l1 norm')
                self.model = prune_model_l1_strucured(self.model, pruning_perc=self.prune_l1_unstructured[1])
                print('done')
            
            if self.prune_torch_pruning[0]:
            
                for param in self.parameters():
                    param.requires_grad = True
                example_inputs = torch.rand((1,3,224,224))
                imp = tp.importance.TaylorImportance()
                ignored_layers = [m for m in self.modules()][-2:]
                iterative_steps = 5 # progressive pruning
                pruner = tp.pruner.MagnitudePruner(
                    self,
                    example_inputs,
                    importance=imp,
                    iterative_steps=iterative_steps,
                    ch_sparsity=self.prune_torch_pruning[1], 
                    ignored_layers=ignored_layers,
                )
                # print(self.model(example_inputs))
                for i in range(iterative_steps):
                    if isinstance(imp, tp.importance.TaylorImportance):
                        # Taylor expansion requires gradients for importance estimation
                        loss = self.model(example_inputs)[0].sum() # a dummy loss for TaylorImportance
                        loss.backward() # before pruner.step()
                    pruner.step()
                for param in self.parameters():
                    param.requires_grad = False
                    
            if int(1) in self.layers_needed:
                list(self.model.children())[4][-1].register_forward_hook(self.hook_t)
            if int(2) in self.layers_needed:
                list(self.model.children())[5][-1].register_forward_hook(self.hook_t)
            if int(3) in self.layers_needed:
                list(self.model.children())[6][-1].register_forward_hook(self.hook_t)
            if int(4) in self.layers_needed:
                list(self.model.children())[7][-1].register_forward_hook(self.hook_t)

        elif not self.layer_cut and not self.need_for_own_last_layer: #take the whole model
            if self.prune_l1_unstructured[0]:
                print('prune l1 norm')
                self.model = prune_model_l1_strucured(self.model, pruning_perc=self.prune_l1_unstructured[1])
                print('done')
            if int(1) in self.layers_needed:
                self.model.layer1[-1].register_forward_hook(self.hook_t)
            if int(2) in self.layers_needed:
                self.model.layer2[-1].register_forward_hook(self.hook_t)
            if int(3) in self.layers_needed:
                self.model.layer3[-1].register_forward_hook(self.hook_t)
            if int(4) in self.layers_needed:
                self.model.layer4[-1].register_forward_hook(self.hook_t)

        elif self.need_for_own_last_layer and self.model_id.__contains__('W'):
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
                output_layer = OwnBottleneck(dict_1, dict_2, dict_3, self.prune_output_layer[0], self.selected_idx_dict[max(self.layers_needed)], input_size, self.exclude_relu, self.sigmoid_in_last_layer)
            else:
                output_layer = OwnBottleneck(dict_1, dict_2, dict_3, self.prune_output_layer[0], self.prune_output_layer[1], input_size, self.exclude_relu, self.sigmoid_in_last_layer)
            del self.model
            self.model = nn.Sequential(layers_1, layers_2, output_layer)

            if self.prune_l1_unstructured[0]:
                print('prune l1 norm')
                self.model = prune_model_l1_strucured(self.model, pruning_perc=self.prune_l1_unstructured[1])
                print('done')
            
            if len(self.layers_needed) > 1:
                for layer in self.layers_needed[:-1]:
                    list(self.model.children())[0][layer+3][-1].register_forward_hook(self.hook_WideResNet)
            self.model[-1].register_forward_hook(self.hook_t)
        
        elif self.need_for_own_last_layer and not self.model_id.__contains__('W'):
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
                output_layer = OwnBasicblock(dict_1, dict_2, self.prune_output_layer[0], self.selected_idx_dict[max(self.layers_needed)], input_size, self.exclude_relu, self.sigmoid_in_last_layer)
            else:
                output_layer = OwnBasicblock(dict_1, dict_2, self.prune_output_layer[0], self.prune_output_layer[1], input_size, self.exclude_relu, self.sigmoid_in_last_layer)  
            del self.model
            self.model = nn.Sequential(layers_1, layers_2, output_layer)

            if self.prune_l1_unstructured[0]:
                print('prune l1 norm')
                self.model = prune_model_l1_strucured(self.model, pruning_perc=self.prune_l1_unstructured[1])
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
        # print(x_t.device)
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

def prune_model_l1_strucured(model, pruning_perc):
    '''
    Prune the model with the given pruning_perc. Decisions are made based on the L1 norm of the weights. 
    '''
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=pruning_perc, n=1, dim=1)
            # prune.remove(module, 'weight')

    return model

def prune_model_nni(model, config_list, method = 'L1', print_logs=False):
    '''
    Prune the model with the given pruning_perc. Decisions are made based on the L1 norm of the weights. Utilizes the nni pruning pipeline by microsoft.
    '''
    if method.__contains__('L1'):
        from nni.compression.pytorch.pruning import L1NormPruner as Pruner #,L1NormPruner #L2NormPruner,
    elif method.__contains__('L2'):
        from nni.compression.pytorch.pruning import L2NormPruner as Pruner
    elif method.__contains__('FPGM'):
        from nni.compression.pytorch.pruning import FPGMPruner as Pruner
    
    # config_list = [{
    #     'op_types': ['Conv2d'],
    #     'total_sparsity': pruning_perc
    # }, {
    #     'exclude': True,
    #     'op_names': ['OwnBasicblock.block_1.final_3']
    # }]
    
    # pruner = L1NormPruner(model, config_list)
    pruner = Pruner(model, config_list) #not working
    # pruner = L2NormPruner(model, config_list)
    
    # compress the model and generate the masks
    _, masks = pruner.compress()
    # show the masks sparsity
    if print_logs:
        for name, mask in masks.items():
            print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))
        
    # need to unwrap the model, if the model is wrapped before speedup
    pruner._unwrap_model()

    # speedup the model, for more information about speedup, please refer :doc:`pruning_speedup`.
    from nni.compression.pytorch.speedup import ModelSpeedup

    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    
    ModelSpeedup(model, torch.rand(1, 3, 224, 224).to(device), masks).speedup_model() # .to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def compress_model_nni(model):#, pruning_perc, method = 'L1', print_logs=False):

    from nni.algorithms.compression.pytorch.quantization import NaiveQuantizer as Compressor
    # model = model.cuda()
    print('model device: ', next(model.parameters()).device)
    config_list = [{
        'quant_types': ['weight'],
        'quant_bits': {'weight': 8},
        'op_types': ['Conv2d']
    }]
    
    model = Compressor(model, config_list).compress()
    
    return model

class OwnBottleneck(torch.nn.Module):
    def __init__(self, block_1, block_2, block_3, prune_output_layer, idx_selected, input_size, exclude_relu = False, sigmoid_in_last_layer=False):
        '''
        just pass OderedDicts, bottleneck like layer is created. For ResNet with Bottlenecks. 
        '''
        super().__init__()
        self.block_1 = torch.nn.Sequential(block_1)
        self.block_2 = torch.nn.Sequential(block_2)
        self.block_3 = torch.nn.Sequential(block_3)
        if exclude_relu:
            self.output_activation = torch.nn.Identity()#inplace=True)
        elif sigmoid_in_last_layer:
            self.output_activation = torch.nn.Sigmoid()#inplace=True)
        else:
            self.output_activation = torch.nn.ReLU(inplace=True)
        self.idx_selected = idx_selected
        self.prune_output_layer = prune_output_layer
        if len(self.idx_selected) > 0:
            channels_not_selected = [i for i in range(input_size[1]*2) if i not in self.idx_selected]
            DG = tp.DependencyGraph().build_dependency(self.block_3, example_inputs=torch.rand(input_size))
            group = DG.get_pruning_group(self.block_3.final_4, tp.prune_conv_out_channels, idxs=channels_not_selected)
            print(group)
            if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.  
                group.prune()
        
    def forward(self, x):
        # identity = x
        if self.prune_output_layer:
            identity = x[:, self.idx_selected, ...]
        else:
            identity = x
        
        out = self.block_1(x)
        out = self.relu(out)
        
        out = self.block_2(out)
        out = self.relu(out)
        
        out = self.block_3(out)
        # identity = identity[:,self.idx_selected,...]
        
        out += identity
        out = self.output_activation(out)
                
        return out
    
class OwnBasicblock(torch.nn.Module):
    def __init__(self, block_1, block_2, prune_output_layer, idx_selected, input_size, exclude_relu = False, sigmoid_in_last_layer=False):
        '''
        just pass OderedDicts, bottleneck like layer is created. For ResNet with Basicblocks. 
        '''
        super().__init__()
        # print(block_1)
        # print(block_2)
        # print(prune_output_layer)
        # print(idx_selected)
        # print(exclude_relu)
        # print(sigmoid_in_last_layer)
        # print(input_size)
        print('HERE')
        self.block_1 = torch.nn.Sequential(block_1)
        self.block_2 = torch.nn.Sequential(block_2)
        if exclude_relu:
            self.output_activation = torch.nn.Identity()#inplace=True)
        elif sigmoid_in_last_layer:
            self.output_activation = torch.nn.Sigmoid()#inplace=True)
        else:
            self.output_activation = torch.nn.ReLU(inplace=True)
        # print(self.block_1)
        # print(self.block_2)
        self.idx_selected = idx_selected
        self.prune_output_layer = prune_output_layer
        if len(self.idx_selected) > 0 and prune_output_layer:
            channels_not_selected = [i for i in range(input_size[1]) if i not in self.idx_selected]
            DG = tp.DependencyGraph().build_dependency(self.block_2, example_inputs=torch.rand(input_size))
            group = DG.get_pruning_group(self.block_2.final_3, tp.prune_conv_out_channels, idxs=channels_not_selected)
            print(group)
            if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.  
                group.prune()
        
    def forward(self, x):
        if self.prune_output_layer:
            identity = x[:, self.idx_selected, ...]
        else:
            identity = x
        out = self.block_1(x)
        out = self.block_2(out)
        # print('out shape: ', out.shape)
        # print('identity shape: ', identity.shape)
        out += identity
        out = self.output_activation(out)
        return out

def prune_naive(model, pruning_perc):
    for param in model.parameters():
        param.requires_grad = True

    # summary(model, input_size=(1,3, 224, 224), verbose=1)
    # Importance criteria
    example_inputs = torch.randn(1, 3, 224, 224).cuda()
    imp = tp.importance.TaylorImportance()

    # ignored_layers = []
    # for m in model.modules():
    #     if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
    #         ignored_layers.append(m) # DO NOT prune the final classifier!
    ignored_layers = [[m for m in model.modules()][-1]]

    iterative_steps = 5 # progressive pruning
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=pruning_perc, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )
    
    # base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):
        if isinstance(imp, tp.importance.TaylorImportance):
            # Taylor expansion requires gradients for importance estimation
            loss = model(example_inputs)[0].sum() # a dummy loss for TaylorImportance
            loss.backward() # before pruner.step()
        pruner.step()
        # macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        # finetune your model here
        # finetune(model)
    for param in model.parameters():
        param.requires_grad = False  
    return model

def prune_output_layer(model, idx_selected, max_index, input_size=(1,3,224,224)):
    '''
    prune the output layer of the model, i.e., the last layer of the model. Removes the channels that are not selected.
    '''

    channels_not_selected = [i for i in range(max_index) if i not in idx_selected] # TODO dynamic 128
    print('channels not selected: ', channels_not_selected)
    device = next(model.model.parameters()).device

    for param in model.model.parameters():
        param.requires_grad = True

    DG = tp.DependencyGraph().build_dependency(model.model, example_inputs=torch.rand(input_size).to(device))
    group = DG.get_pruning_group(model.model[2].block_2.final_3, tp.prune_conv_out_channels, idxs=channels_not_selected)
    print(group)
    if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.  
        group.prune()
        
    for param in model.model.parameters():
        param.requires_grad = False

    return model