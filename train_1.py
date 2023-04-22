import os
import glob
import shutil

import numpy as np
import pandas as pd
from PIL import Image
import cv2

from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import roc_auc_score
from sampling_methods.kcenter_greedy import kCenterGreedy
from scipy.ndimage import gaussian_filter
import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import faiss
import pickle
from sklearn.neighbors import NearestNeighbors
import time
import math
from time import perf_counter as record_cpu
from anomalib.models.components.sampling import k_center_greedy
import torch_pruning as tp
from torchinfo import summary
from collections import OrderedDict


def record_gpu(cuda_event):
    '''
    gpu_measurement
    '''
    cuda_event.record()
    torch.cuda.synchronize()
    
    return cuda_event

def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)
    return dist


class NN():
    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):
    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        dist = torch.cdist(x, self.train_pts, self.p)
        knn = dist.topk(self.k, largest=False)
        return knn

class OwnBottleneck(torch.nn.Module):
    def __init__(self, block_1, block_2, block_3, idx_selected, input_size):
        '''
        just pass OderedDicts, bottleneck like layer is created.
        '''
        super().__init__()
        self.block_1 = torch.nn.Sequential(block_1)
        self.block_2 = torch.nn.Sequential(block_2)
        self.block_3 = torch.nn.Sequential(block_3)
        self.relu = torch.nn.ReLU(inplace=True)
        self.idx_selected = idx_selected
        channels_not_selected = [i for i in range(input_size[1]*2) if i not in idx_selected]
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
        out = self.relu(out)
                
        return out

def prep_dirs(root):
    # make embeddings dir
    embeddings_path = os.path.join('./', 'embeddings', args.category)
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    return embeddings_path, sample_path, source_code_save_path

def embedding_concat(x, y):
    '''
    alligns dimensions
    
    TODO: numba version plus lightweight version
    
    from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    '''
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z

def reshape_embedding(embedding):
    '''
    flattens spatial dimensions and concatenates channels. Results in 1D-Vector
    
    TODO: numba or numpy version! 
    '''
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

class MVTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    
    

class PatchCore(pl.LightningModule):
    def __init__(self, hparams):
        super(PatchCore, self).__init__()
        
        # options
        self.faiss = True # temp
        self.quantization = False
        self.measure_inference = True
        self.number_of_reps = 50 # number of reps during measurement. Beacause we can assume a consistent estimator, results get more accurate with more reps
        self.warm_up_reps = 10 # before the actual measurement is done, we execute the process a couple of times without measurement to ensure that there is no influence of initialization and that the circumstances (e.g. thermal state of hardware) are representive.
        self.cuda_active = torch.cuda.is_available()
        self.dim_reduction = False
        self.log_file_name = f'trial_{int(time.time())}.csv'
        self.save_am = False
        self.only_img_lvl = True
        self.save_features = False
        if self.save_features:
            self.features_to_store = []
        self.save_embeddings = False
        self.reduce_via_std = False
        self.reduce_via_entropy = False
        
        self.pruning = False
        
        self.save_hyperparameters(hparams)

        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)

        # backbone selection   
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
        
        for param in self.model.parameters():
            param.requires_grad = False

        # feature map selection
        self.model.layer2[-1].register_forward_hook(hook_t)
        # self.model.layer3[-1].register_forward_hook(hook_t)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.init_results_list()

        self.data_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size)])

        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])
        
        if self.quantization:
            self = self.half()

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []        

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def save_anomaly_map(self, anomaly_map, input_img, gt_img, file_name, x_type):
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm*255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)

    def train_dataloader(self):
        image_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0)
        return train_loader

    def test_dataloader(self):
        test_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0)
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        self.model.eval() # to stop running_var move (maybe not critical)        
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.embedding_list = []
    
    def on_test_start(self):
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        if self.faiss:
            self.index = faiss.read_index(os.path.join(self.embedding_dir_path,'index.faiss'))
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
        elif False:
            self.knn = KNN(torch.from_numpy(self.embedding_coreset).cuda(), k=9)
        else:
            self.nbrs = NearestNeighbors(n_neighbors=args.n_neighbors, algorithm='ball_tree', metric='minkowski', p=2).fit(self.embedding_coreset)
        self.init_results_list()
        
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, _, _, _ = batch
        features = self(x)
        if self.save_features: # only one layer at a time!!
            self.features_to_store.append(features[0].detach().cpu())        
        embeddings = []
        for k, feature in enumerate(features):
            pooled_feature = torch.nn.AvgPool2d(3, 1, 1)(feature)#self.adaptive_pooling(feature)# using AvgPool2d to calculate local-aware features
            # if k in args.feature_map_to_reduce and args.partial_reduction:
            #     org_shape = pooled_feature.shape
            #     pooled_feature = self.partial_reducer.transform(np.reshape(pooled_feature.cpu(), (-1, org_shape[1])))
            #     pooled_feature = torch.from_numpy(np.reshape(pooled_feature, (org_shape[0],-1, org_shape[2], org_shape[3])))
            #     if not pooled_feature.device.__str__().__contains__(self.accelerator):
            #         pooled_feature = pooled_feature.to(self.accelerator)
            embeddings.append(pooled_feature)
        embedding = self.embedding_concat_frame(embeddings=embeddings) # shape (batch, 448, 16, 16) --> default
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))
            
    def training_epoch_end(self, outputs):
        if self.save_features:
            file_name_features = input('file name for features:\n')
            # feature_save = np.array([])
            for k1, el in enumerate(self.features_to_store):
                for k2, l in enumerate(el):
                    if k1 == 0 and k2 == 0:
                        feature_save = np.expand_dims(l.cpu().numpy(), axis=0)
                    feature_save = np.append(feature_save, np.expand_dims(l.cpu().numpy(), axis=0), axis=0)
            print(feature_save.shape)
            np.save(file_name_features + '.npy', feature_save)
        total_embeddings = np.array(self.embedding_list)
        
        if self.reduce_via_std:
            percentile_std = 50
            org_no_channels = total_embeddings.shape[1] # total_embeddings = (200000, 512)
            self.idx_chosen = np.argwhere(np.std(total_embeddings, axis=0)>np.percentile(np.std(total_embeddings,axis=0), percentile_std))[:,0]
            total_embeddings = np.take(total_embeddings, self.idx_chosen, axis=1)#total_embeddings[:,self.idx_with_high_std] # c contigous
        
        if self.reduce_via_entropy:
            percentile_entropy = 90
            org_no_channels = total_embeddings.shape[1]
            total_embeddings_copy = total_embeddings.copy()
            total_embeddings_copy[total_embeddings_copy<1e-15] = 1e-15
            entropy = -np.sum(total_embeddings_copy*np.log2(total_embeddings_copy), axis=0)#.shape
            self.idx_chosen = np.argwhere(entropy>np.percentile(entropy, percentile_entropy))[:,0]
            total_embeddings = np.take(total_embeddings, self.idx_chosen, axis=1)
        if self.save_embeddings:
            file_name_embeddings = input('file name for embeddings:\n')
            np.save(file_name_embeddings + '.npy', total_embeddings)
        if self.pruning and (self.reduce_via_entropy or self.reduce_via_std):
            print('Pruning ...')

            # import copy
            
            print('full net:') # TODO: Replace with args
            summary(self.model, depth=5, input_size=(1,3,224,224), col_names=['input_size', 'output_size', 'trainable', 'mult_adds', 'num_params'])
            model_full = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)#self.model.copy()
            layer_to_include = 2 #TODO
            # model_cutted = torch.nn.Sequential(*(list(model_full.children())[0:int(4+layer_to_include)])) # TODO: Replace with args
            # print('\n\nnet cutted:')
            # summary(model_cutted, input_size=(1,3,224,224), depth=3, verbose=1)# TODO: Replace with args
            # model_cutted.to('cpu')
            # DG = tp.DependencyGraph().build_dependency(model_cutted, example_inputs=torch.randn(1,3,224,224)) # TODO: Replace with args
            # channels_not_selected = [i for i in range(org_no_channels) if i not in self.idx_chosen]
            # print(model_cutted[5][3].conv3)
            # group = DG.get_pruning_group(model_cutted[5][3].conv3, tp.prune_conv_out_channels, idxs=channels_not_selected) # take last conv of 2nd Layer # TODO dynamic!
            # if DG.check_pruning_group(group):
            #     group.prune()
            
            # print('\n\nfully pruned:')
            # summary(model_cutted, input_size=(1,3,224,224), depth=3, verbose=1)
            # this_model = torch.nn.Sequential(*(list(model_full.children())[:int(4+layer_to_include)]))
            # DG = tp.DependencyGraph().build_dependency(this_model, example_inputs=torch.randn(1,3,224,224))
            # group = DG.get_pruning_group(this_model[-1][-1].conv3, tp.prune_conv_out_channels, idxs=channels_not_selected)
            model_1st = torch.nn.Sequential(*(list(model_full.children())[:int(4+layer_to_include-1)]))
            model_2nd_layer_1 = torch.nn.Sequential(*(list(model_full.children())[int(4+layer_to_include-1)][:-1]))
            model_2nd_layer_2_tmp = torch.nn.Sequential(*(list(model_full.children())[int(4+layer_to_include-1)][-1:]))
            # last_layer_orderered_dict = OrderedDict([(f'final_{i}', module) for i, module in enumerate(model_2nd_layer_2_tmp[0].modules()) if i != 0])
            # last_layer = torch.nn.Sequential(last_layer_orderered_dict)
            dict_1 = OrderedDict([(f'final_{i}', module) for i, module in enumerate(model_2nd_layer_2_tmp[-1].children()) if i<2])
            dict_2 = OrderedDict([(f'final_{i}', module) for i, module in enumerate(model_2nd_layer_2_tmp[-1].children()) if i<4 and i>=2])
            dict_3 = OrderedDict([(f'final_{i}', module) for i, module in enumerate(model_2nd_layer_2_tmp[-1].children()) if i<6 and i>=4])
            print(dict_3)
            input_size = (1,256,28,28) #TODO dynamicalyy
            new_layer = OwnBottleneck(dict_1, dict_2, dict_3, self.idx_chosen, input_size)
            # this_model = torch.nn.Sequential(*(list(model_full.children())[0:int(4+layer_to_include)]))

            # DG = tp.DependencyGraph().build_dependency(new_layer, example_inputs=torch.randn(1,512,28,28)) # TODO

            # # 2. Specify the to-be-pruned channels. Here we prune those channels indexed by [2, 6, 9].
            # group = DG.get_pruning_group(new_layer.block_3[-1], tp.prune_batchnorm_out_channels, idxs=channels_not_selected)
            # print(group)
            # # 3. prune all grouped layers that are coupled with model.conv1 (included).
            # if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.
            #     # print('hey')
                # group.prune()
            # self.model = this_model
            # group = DG.get_pruning_group(model_final[2].final_5, tp.prune_batchnorm_in_channels, idxs=channels_not_selected)
            # # 3. prune all grouped layers that are coupled with model.conv1 (included).
            # print(group)
            # if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.
            #     # print('hey')
            #     group.prune()
            # group = DG.get_pruning_group(model_final[2].final_5, tp.prune_batchnorm_out_channels, idxs=channels_not_selected)
            # print(group)
            # if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.
            #     print('hey')
            #     group.prune()
            # model_final.to('cuda') #TODO
            # del self.model
            # self.model = model_final.to('cuda') #TODO
            del self.model
            self.model = torch.nn.Sequential(model_1st, model_2nd_layer_1, new_layer)
            summary(self.model, depth=5, input_size=(1,3,224,224), col_names=['input_size', 'output_size', 'trainable', 'mult_adds', 'num_params'])
            # self.model = this_model
            self.init_features()
            def hook_t(module, input, output):
                self.features.append(output)
        
            for param in self.model.parameters():
                param.requires_grad = False

            # # feature map selection
            self.model[-1].register_forward_hook(hook_t) #
            # self.model.layer3[-1].register_forward_hook(hook_t)

            # self.criterion = torch.nn.MSELoss(reduction='sum')

            # self.init_results_list()
                
        # Random projection
        if args.coreset_sampling_ratio == 1.0:
            self.embedding_coreset = total_embeddings
        else:                  
            if False: # two different implementation that yield the same result (approximately)
                self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
                self.randomprojector.fit(total_embeddings)
                # Coreset Subsampling
                selector = kCenterGreedy(total_embeddings,0,0)
                selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*args.coreset_sampling_ratio))
            else:
                sampler = k_center_greedy.KCenterGreedy(embedding=torch.from_numpy(total_embeddings), sampling_ratio=float(args.coreset_sampling_ratio))
                selected_idx = sampler.select_coreset_idxs()
            
            self.embedding_coreset = total_embeddings[selected_idx]
            
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        #faiss
        if self.faiss:
            self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
            self.index.add(self.embedding_coreset) 
            faiss.write_index(self.index,  os.path.join(self.embedding_dir_path,'index.faiss'))
        else:
            with open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'wb') as f:
                pickle.dump(self.embedding_coreset, f)
            
    def test_step(self, batch, batch_idx):
        '''
        required func that handles not just the step istelf, but also the measurements (inference times).
        '''
        if self.measure_inference:
            # initialize dict
            run_times = {
                    '#1 feature extraction cpu': [],
                    '#2 feature extraction gpu': [],
                    '#3 embedding of features cpu': [],
                    '#4 embedding of features gpu': [],
                    '#5 score patches cpu': [],
                    '#6 score patches gpu': [],
                    '#7 img lvl score cpu': [],
                    '#8 img lvl score gpu': [],
                    '#9 anomaly map cpu': [],
                    '#10 anomaly map gpu': [],
                    '#11 whole process cpu': [],
                    '#12 whole process gpu': [],
                    '#13 dim reduction cpu': [],
                    '#14 dim reduction gpu': []          
                }
            # warm up loop
            for _ in range(self.warm_up_reps):
                _, _, _, _, _ = self.test_step_core(batch=batch, measure=False)
            
            # actual measurements
            ################################################
            # LOOP
            for rep in range(self.number_of_reps):
                if self.cuda_active:
                    st_gpu, et_gpu = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True) # initialize cuda timers
                    st_gpu = record_gpu(st_gpu)
                st_cpu = record_cpu()
                if rep+1 == self.number_of_reps:
                    features, embeddings, score_patches, score, anomaly_map, t_0_cpu, t_1_cpu, t_2_cpu, t_3_cpu, t_4_cpu, t_0_gpu, t_1_gpu, t_2_gpu, t_3_gpu, t_4_gpu = self.test_step_core(batch=batch, measure=True)
                else:
                    _, _, _, _, _, t_0_cpu, t_1_cpu, t_2_cpu, t_3_cpu, t_4_cpu, t_0_gpu, t_1_gpu, t_2_gpu, t_3_gpu, t_4_gpu = self.test_step_core(batch=batch, measure=True)
                et_cpu = record_cpu()
                # gpu
                if self.cuda_active:
                    et_gpu = record_gpu(et_gpu)
                    run_times['#2 feature extraction gpu'] += [t_0_gpu.elapsed_time(t_1_gpu)]
                    run_times['#4 embedding of features gpu'] += [t_1_gpu.elapsed_time(t_2_gpu)]
                    run_times['#6 score patches gpu'] += [t_2_gpu.elapsed_time(t_3_gpu)]
                    run_times['#8 img lvl score gpu'] += [t_3_gpu.elapsed_time(t_4_gpu)]
                    run_times['#10 anomaly map gpu'] += [t_4_gpu.elapsed_time(et_gpu)]
                    run_times['#12 whole process gpu'] += [st_gpu.elapsed_time(et_gpu)]
                    if self.dim_reduction:
                        run_times['#14 dim reduction gpu'] += [100.0] # TODO
                    else:
                        run_times['#14 dim reduction gpu'] += [0.0]
                else:
                    run_times['#2 feature extraction gpu'] += [0.0]
                    run_times['#4 embedding of features gpu'] += [0.0]
                    run_times['#6 score patches gpu'] += [0.0]
                    run_times['#8 img lvl score gpu'] += [0.0]
                    run_times['#10 anomaly map gpu'] += [0.0]
                    run_times['#12 whole process gpu'] += [0.0]
                    run_times['#14 dim reduction gpu'] += [0.0]
                # cpu
                run_times['#1 feature extraction cpu'] += [float((t_1_cpu - t_0_cpu) * 1e3)]
                run_times['#3 embedding of features cpu'] += [float((t_2_cpu - t_1_cpu) * 1e3)]
                run_times['#5 score patches cpu'] += [float((t_3_cpu - t_2_cpu) * 1e3)]
                run_times['#7 img lvl score cpu'] += [float((t_4_cpu - t_3_cpu) * 1e3)]
                run_times['#9 anomaly map cpu'] += [float((et_cpu - t_4_cpu) * 1e3)]
                run_times['#11 whole process cpu'] += [float((et_cpu - st_cpu) * 1e3)]
                if self.dim_reduction:
                    run_times['#13 dim reduction cpu'] += [100.0] # TODO
                else:
                    run_times['#13 dim reduction cpu'] += [0.0]
            # LOOP
            ################################################
           
            assert len(run_times['#1 feature extraction cpu']) == self.number_of_reps, "Something went wrong!"
            
            # calc mean of measurements
            for this_entry in run_times.items():
                if len(this_entry[1]) > 0:
                    run_times[this_entry[0]] = float((sum(this_entry[1]) / len(this_entry[1])) / batch[0].size()[0]) # mean
                else:
                    run_times[this_entry[0]] = 0.0
            
            # note args used for this run and add them to dict
            # TODO
            
            # save as csv using pandas dataframe
            pd_run_times = pd.DataFrame(run_times, index=[batch_idx])
            if os.path.exists(os.path.join(os.path.dirname(__file__), "results","csv", self.log_file_name)):
                pd_run_times_ = pd.read_csv(os.path.join(os.path.dirname(__file__), "results", "csv",self.log_file_name), index_col=0)
                pd_run_times = pd.concat([pd_run_times_, pd_run_times], axis=0)
                pd_run_times.to_csv(os.path.join(os.path.dirname(__file__), "results","csv", self.log_file_name))
            else:
                pd_run_times.to_csv(os.path.join(os.path.dirname(__file__), "results","csv", self.log_file_name))
        
        else:
            _, _, score_patches, score, anomaly_map = self.test_step_core(batch=batch, measure=False)
                # calculating of scores and saving of results
        
        if type(score_patches) == list:
            results = (score_patches, anomaly_map)
            x_batch, gt_batch, label_batch, file_name_batch, x_type_batch = batch
            for k in range(x_batch.size()[0]):
                this_score_patches, this_anomaly_map = results[0][k], results[1][k]
                x, gt, label, file_name, x_type = x_batch[k], gt_batch[k], label_batch[k], file_name_batch[k], x_type_batch[k]
                self.eval_one_step_test(score_patches=this_score_patches, score=score, anomaly_map=this_anomaly_map, x=x, gt=gt, label=label, file_name=file_name, x_type=x_type)
        else:
            x, gt, label, file_name, x_type = batch
            self.eval_one_step_test(score_patches, score, anomaly_map, x, gt, label, file_name, x_type)
            
                               
    def test_step_core(self, batch, measure=False):
        '''
        basically this is one test step where one batch is processed. This func is embedded in the actual def test_step. 
        ''' 
        if not measure:
            x, _, _, _, _ = batch
            batch_size_1 = (x.shape[0] == 1)
            batch_size = x.shape[0]
            # extract embedding
            features = self.feature_extraction(x=x)
            embeddings = self.feature_embedding(features=features, batch_size_1=batch_size_1, batch_size=batch_size)
            score_patches = self.calc_score_patches(embeddings=embeddings, batch_size_1=batch_size_1)
            score = self.calc_img_score(score_patches=score_patches)
            if not self.only_img_lvl:
                anomaly_map = self.calc_anomaly_map(score_patches=score_patches, batch_size_1=batch_size_1)
            else:
                anomaly_map = None
                
            return features, embeddings, score_patches, score, anomaly_map 
            
        else:
            ############################################################
            # INITIALIZE MEASUREMENT UTILS
            # initialize cuda events
            if torch.cuda.is_available():
                t_0_gpu, t_1_gpu, t_2_gpu, t_3_gpu, t_4_gpu = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            else:
                t_0_gpu, t_1_gpu, t_2_gpu, t_3_gpu, t_4_gpu = None, None, None, None, None
            # INITIALIZE MEASUREMENT UTILS
            ############################################################
            
            ############################################################
            # FEATURE EXTRACTION
            t_0_cpu = record_cpu()
            if self.cuda_active:
                t_0_gpu = record_gpu(t_0_gpu)
            
            x, _, _, _, _ = batch
            batch_size_1 = (x.shape[0] == 1)
            batch_size = x.shape[0]
            
            features = self.feature_extraction(x=x)
            # FEATURE EXTRACTION
            ############################################################

            ############################################################            
            # FEATURE EMBEDDING
            t_1_cpu = record_cpu()
            if self.cuda_active:
                t_1_gpu = record_gpu(t_1_gpu)
                        
            embeddings = self.feature_embedding(features=features, batch_size_1=batch_size_1, batch_size=batch_size)
            # FEATURE EMBEDDING
            ############################################################
            
            ############################################################
            # NN SEARCH // SCORE PATCHES
            t_2_cpu = record_cpu()
            t_2_gpu = record_gpu(t_2_gpu)
            
            score_patches = self.calc_score_patches(embeddings=embeddings, batch_size_1=batch_size_1)
            # NN SEARCH // SCORE PATCHES
            ############################################################
            
            ############################################################
            # IMG LEVEL SCORE
            t_3_cpu = record_cpu()
            t_3_gpu = record_gpu(t_3_gpu)

            score = self.calc_img_score(score_patches=score_patches)
            # IMG LEVEL SCORE
            ############################################################
            
            ############################################################
            # AMOMALY MAP
            t_4_cpu = record_cpu()
            t_4_gpu = record_gpu(t_4_gpu)
            if not self.only_img_lvl:
                anomaly_map = self.calc_anomaly_map(score_patches=score_patches, batch_size_1=batch_size_1)
            else:
                anomaly_map = None
            # ANOMALY MAP
            ############################################################
            
            return features, embeddings, score_patches, score, anomaly_map, t_0_cpu, t_1_cpu, t_2_cpu, t_3_cpu, t_4_cpu, t_0_gpu, t_1_gpu, t_2_gpu, t_3_gpu, t_4_gpu
            
    def feature_extraction(self, x):
        '''
        Pass data through backbone specified in class pactchcore
        '''
        if self.cuda_active:
            x = x.cuda()
        if False:#self.pruning and (self.reduce_via_entropy or self.reduce_via_std): # TODO
            # output = self(x)
            # output = np.array(output[0].cpu())
            # return torch.from_numpy(np.take(output, self.idx_chosen, axis=1))
            return [self(x)[0][:,self.idx_chosen,:,:]]
        else:
            return self(x)
            
    
    def feature_embedding(self, features, batch_size_1, batch_size):
        '''
        embedding of features extracted in previous step. Eventually integrates dim reduction and adaptive pooling. 
        '''
        selected_features = []
            
        for no_feature_map, feature in enumerate(features):
            ####
            # insert dim reduction here TODO 
            # before pooling
            ####
            pooled_features = torch.nn.AvgPool2d(3, 1, 1)(feature) # TODO replace with adaptive pooling
            ####
            # insert dim reduction here TODO 
            # after pooling
            ####
            selected_features.append(pooled_features)
        
        concatenated_features = self.embedding_concat_frame(embeddings=selected_features)
            
        
        if batch_size_1:
            flattened_features = np.array(reshape_embedding(np.array(concatenated_features)))
        else:
            flattened_features = np.array([np.array(reshape_embedding(np.array(concatenated_features[k,...].unsqueeze(0)))) for k in range(batch_size)])
        
        if (self.reduce_via_std or self.reduce_via_entropy) and not self.pruning:
            return np.take(flattened_features, self.idx_chosen, axis=1)#indices=#[:,self.idx_with_high_std]
        else:
            return flattened_features
        
    def calc_score_patches(self, embeddings, batch_size_1):
        '''
        calc score_patches from which image score and anomaly map can be derived.
        '''
        if batch_size_1:
            if self.faiss:
                score_patches, _ = self.index.search(embeddings , k=args.n_neighbors)
            elif False:
                score_patches = self.knn(torch.from_numpy(embeddings).cuda())[0].cpu().detach().numpy()
            else:
                score_patches, _ = self.nbrs.kneighbors(embeddings)
        else:
            if self.faiss:
                score_patches = [self.index.search(element, k=args.n_neighbors)[0] for element in embeddings]
            elif False:
                score_patches = [self.knn(torch.from_numpy(element).cuda()[0].cpu().detach().numpy()) for element in embeddings]
            else:
                score_patches = [self.nbrs.kneighbors(element) for element in embeddings]
        
        return score_patches

    def calc_img_score(self, score_patches):
        '''
        calculates the image score based on score_patches
        '''
        N_b = score_patches[np.argmax(score_patches[:,0])]
        w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
        score = w*max(score_patches[:,0]) # Image-level score #TODO --> meaning of numbers
        return score
    
    def calc_anomaly_map(self, score_patches, batch_size_1):
        '''
        calculates anomaly map based on score_patches
        '''
        if batch_size_1:
            anomaly_map = score_patches[:,0].reshape((int(math.sqrt(len(score_patches[:,0]))),int(math.sqrt(len(score_patches[:,0])))))
            a = int(args.load_size) # int, 64 
            anomaly_map_resized = cv2.resize(anomaly_map, (a, a)) # [8,8] --> [64,64]
            anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)# shape [8,8]
        else:
            anomaly_map = [score_patch[:,0].reshape((int(math.sqrt(len(score_patch[:,0]))),int(math.sqrt(len(score_patch[:,0]))))) for score_patch in score_patches]
            a = int(args.load_size)
            anomaly_map_resized = [cv2.resize(this_anomaly_map, (a, a)) for this_anomaly_map in anomaly_map]
            anomaly_map_resized_blur = [gaussian_filter(this_anomaly_map_resized, sigma=4) for this_anomaly_map_resized in anomaly_map_resized]
        return anomaly_map_resized_blur
    
    def eval_one_step_test(self, score_patches, score, anomaly_map, x, gt, label, file_name, x_type):
        '''
        Extracted evaluation of single output
        '''
        if x.dim() != 4:
            x, gt, label = x.unsqueeze(0), gt.unsqueeze(0), label.unsqueeze(0)
        
        if not self.only_img_lvl:
            gt_np = gt.cpu().numpy()[0,0].astype(int)
            self.gt_list_px_lvl.extend(gt_np.ravel()) # ravel equivalent reshape(-1); flattening of ground_truth pixel wise
            self.pred_list_px_lvl.extend(anomaly_map.ravel()) # flattening of pred pixel wise
        self.gt_list_img_lvl.append(label.cpu().numpy()[0]) # ground_truth for image wise
        self.pred_list_img_lvl.append(score) # image level score appended
        self.img_path_list.extend(file_name) # same for file_name
        # save images
        if self.save_am:
            x = self.inv_normalize(x) # inverse transformation of img
            if x.dtype != torch.float32:
                x = x.to(torch.float32)
            input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB) # further transformation
            self.save_anomaly_map(anomaly_map, input_x, gt_np*255, file_name[0], x_type[0]) # save of everything
        
    def test_epoch_end(self, outputs):
        if not self.only_img_lvl:
            print("Total pixel-level auc-roc score :")
            pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
            print(pixel_auc)
        else:
            pixel_auc = 0.0
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        print('test_epoch_end')
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        self.log_dict(values)
        if os.path.exists(os.path.join(os.path.dirname(__file__), "results","csv", self.log_file_name)) and self.measure_inference:
            pd_run_times_ = pd.read_csv(os.path.join(os.path.dirname(__file__), "results", "csv",self.log_file_name), index_col=0)
            pd_results = pd.DataFrame({'img_auc': [img_auc]*pd_run_times_.shape[0], 'pixel_auc': [pixel_auc]*pd_run_times_.shape[0]})
            pd_run_times = pd.concat([pd_run_times_, pd_results], axis=1)
            pd_run_times.to_csv(os.path.join(os.path.dirname(__file__), "results", "csv",self.log_file_name))
            print(f'\n\nMEAN INFERENCE TIME: {pd_run_times["#11 whole process cpu"].mean()} ms\n')

    def embedding_concat_frame(self, embeddings):
        '''
        framework for concatenating more than two features or less than two
        '''
        no_of_embeddings = len(embeddings)
        if no_of_embeddings == int(1):
            embeddings_result = embeddings[0].cpu()
        elif no_of_embeddings == int(2):
            embeddings_result = embedding_concat(embeddings[0], embeddings[1])
        elif no_of_embeddings > int(2):
            for k in range(no_of_embeddings - 1):
                if k == int(0):
                    embeddings_result = embedding_concat(embeddings[0], embeddings[1]) # default
                    pass
                else:
                    if torch.cuda.is_available() and self.accelerator.__contains__("cuda"):
                        embeddings_result = embedding_concat(embeddings_result.cuda(), embeddings[k+1])
                    else:
                        embeddings_result = embedding_concat(embeddings_result, embeddings[k+1].cpu())
        return embeddings_result

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'/mnt/crucial/UNI/IIIT_Muen/MA/MVTechAD')
    parser.add_argument('--category', default='own')
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=224)
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--coreset_sampling_ratio', default=1.0)
    parser.add_argument('--project_root_path', default=r'./test')
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=1)
    model = PatchCore(hparams=args)
    if args.phase == 'train':
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        trainer.test(model)

