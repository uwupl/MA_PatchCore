import os
from backbone import Backbone
from datasets import MVTecDataset
from utils import min_max_norm, heatmap_on_image, cvt2heatmap, distance_matrix, record_gpu, modified_kNN_score_calc, prep_dirs
from pooling import adaptive_pooling
from embedding import reshape_embedding, embedding_concat_frame
from search import KNN, NN
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
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import faiss
import pickle
from sklearn.neighbors import NearestNeighbors
import time
import math
from time import perf_counter as record_cpu
from anomalib.models.components.sampling import k_center_greedy
from torchinfo import summary

class PatchCore(pl.LightningModule):
    def __init__(self, hparams):
        super(PatchCore, self).__init__()
        
        # options
        self.faiss = False # temp
        self.adapted_score_calc = True
        self.own_knn = True
        self.normalize = True
        self.quantization = False
        self.measure_inference = False
        # self.multiple_filters = ()
        self.number_of_reps = 50 # number of reps during measurement. Beacause we can assume a consistent estimator, results get more accurate with more reps
        self.warm_up_reps = 10 # before the actual measurement is done, we execute the process a couple of times without measurement to ensure that there is no influence of initialization and that the circumstances (e.g. thermal state of hardware) are representive.
        self.cuda_active = False#torch.cuda.is_available()
        self.cuda_active_training = False
        self.dim_reduction = False
        self.log_file_name = f'trial_{int(time.time())}.csv'
        self.save_am = False
        self.only_img_lvl = True
        self.save_features = False
        if self.save_features:
            self.features_to_store = []
        self.save_embeddings = False
        self.reduce_via_std = False
        self.reduce_via_entropy = True
        self.reduce_via_entropy_normed = False
        self.pooling_strategy = ['first_trial']#, 'max_1']#, 'first_trial']#, 'first_trial_max'] # 'first_trial'
        
        self.save_hyperparameters(hparams)
        
        self.model_id = "RN18"
        self.layers_needed = [2]#,3]#,3]
        self.layer_cut = True
        self.prune_output_layer = (False, [])
        
        self.model = Backbone(model_id=self.model_id, layers_needed=self.layers_needed, layer_cut=self.layer_cut, prune_output_layer=(False, []))
        if self.quantization:
            self.model = self.model.half()

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.init_results_list()

        self.data_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])]) # from imagenet
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
        image_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train', half=self.quantization)
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=6)
        return train_loader

    def test_dataloader(self):
        test_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test', half=self.quantization)
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0)
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        self.model.eval() # to stop running_var move (maybe not critical)        
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir, args.category)
        self.embedding_np = np.array([])
    
    def on_test_start(self):
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir, args.category)
        if self.faiss:
            self.index = faiss.read_index(os.path.join(self.embedding_dir_path,'index.faiss'))
            if self.cuda_active:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
        elif self.own_knn:
            self.knn = KNN(torch.from_numpy(self.embedding_coreset), k=args.n_neighbors) #.cuda()
        else:
            self.nbrs = NearestNeighbors(n_neighbors=args.n_neighbors, algorithm='ball_tree', metric='minkowski', p=2).fit(self.embedding_coreset)
        self.init_results_list()
        
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, _, _, _ = batch
        features = self.model(x)
        if self.save_features: # only one layer at a time!!
            self.features_to_store.append(features[0].detach().cpu())        
        embeddings = []
        for k, feature in enumerate(features):
            if type(self.pooling_strategy) == list:
                for strategy in self.pooling_strategy:
                    pooled_feature = adaptive_pooling(feature, strategy)
                    embeddings.append(pooled_feature)
            else:
                pooled_feature = adaptive_pooling(feature, self.pooling_strategy)
                embeddings.append(pooled_feature)
            
        embedding = embedding_concat_frame(embeddings=embeddings, cuda_active=self.cuda_active) # shape (batch, 448, 16, 16) --> default
        if batch_idx == int(0):
            self.embedding_np = reshape_embedding(np.array(embedding))
        else:
            self.embedding_np = np.append(self.embedding_np, reshape_embedding(np.array(embedding)), axis=0)#.extend(reshape_embedding(np.array(embedding)))
            
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
        total_embeddings = self.embedding_np

        if self.reduce_via_std:
            percentile_std = 12.5*2
            self.idx_chosen = np.argwhere(np.std(total_embeddings, axis=0)<np.percentile(np.std(total_embeddings,axis=0), percentile_std))[:,0]
            total_embeddings = np.take(total_embeddings, self.idx_chosen, axis=1)#total_embeddings[:,self.idx_with_high_std] # c contigous
        if self.normalize:
            self.mean = np.mean(total_embeddings, axis=0)
            self.std = np.std(total_embeddings, axis=0)
            total_embeddings = (total_embeddings-self.mean)/self.std
        if self.reduce_via_entropy:
            percentile_entropy = 100-12.5*2*2
            total_embeddings_copy = total_embeddings.copy()
            total_embeddings_copy[total_embeddings_copy<1e-15] = 1e-15
            entropy = -np.sum(total_embeddings_copy*np.log2(total_embeddings_copy), axis=0)#.shape
            self.idx_chosen = np.argwhere(entropy>np.percentile(entropy, percentile_entropy))[:,0]
            total_embeddings = np.take(total_embeddings, self.idx_chosen, axis=1)
        elif self.reduce_via_entropy_normed:
            percentile_entropy = 100-12.5*2*2
            total_embeddings_copy = total_embeddings.copy()
            total_embeddings_copy[total_embeddings_copy<1e-15] = 1e-15
            normed_embeddings = total_embeddings_copy/total_embeddings_copy.sum(axis=1, keepdims=1)
            entropy = -np.sum(normed_embeddings*np.log2(normed_embeddings), axis=0)#.shape
            self.idx_chosen = np.argwhere(entropy>np.percentile(entropy, percentile_entropy))[:,0]
            total_embeddings = np.take(total_embeddings, self.idx_chosen, axis=1)
        if (self.reduce_via_entropy or self.reduce_via_entropy_normed) and self.normalize and not self.reduce_via_std:
            self.std = np.take(self.std, self.idx_chosen)#, axis=0)
            self.mean = np.take(self.mean, self.idx_chosen)#, axis=0)
        if self.save_embeddings:
            file_name_embeddings = input('file name for embeddings:\n')
            np.save(file_name_embeddings + '.npy', total_embeddings)
        if self.prune_output_layer[0] and (self.reduce_via_entropy or self.reduce_via_std or self.reduce_via_entropy_normed):
            print('Pruning ...')        
            self.prune_output_layer = (True, self.idx_chosen)
            self.model = Backbone(model_id=self.model_id, layers_needed=self.layers_needed, layer_cut=self.layer_cut, prune_output_layer=self.prune_output_layer)
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
                # total_embeddings_copy = total_embeddings.astype(np.float32)
                if self.cuda_active or self.cuda_active_training:
                    sampler = k_center_greedy.KCenterGreedy(embedding=torch.from_numpy(total_embeddings).cuda(), sampling_ratio=float(args.coreset_sampling_ratio))
                else:
                    sampler = k_center_greedy.KCenterGreedy(embedding=torch.from_numpy(total_embeddings), sampling_ratio=float(args.coreset_sampling_ratio))
                selected_idx = sampler.select_coreset_idxs()
            
            self.embedding_coreset = total_embeddings[selected_idx]
        # summary(self.model, depth=5, input_size=(1,3,224,224), col_names=['input_size', 'output_size', 'trainable', 'mult_adds', 'num_params'])   
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        #faiss
        if self.faiss:
            if False:
                nlist = 20 if self.embedding_coreset.shape[0] > 20 else self.embedding_coreset.shape[0]
                n_probe = 5 # defaul 1
                quantizer = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_coreset.shape[1], nlist, faiss.METRIC_L2)
                assert not self.index.is_trained
                self.index.train(self.embedding_coreset)
                assert self.index.is_trained
                self.index.add(self.embedding_coreset)
                self.index.nprobe = n_probe
                faiss.write_index(self.index,  os.path.join(self.embedding_dir_path,'index.faiss'))
            else:
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
            if self.cuda_active:
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
            if self.cuda_active:
                t_2_gpu = record_gpu(t_2_gpu)
            
            score_patches = self.calc_score_patches(embeddings=embeddings, batch_size_1=batch_size_1)
            # NN SEARCH // SCORE PATCHES
            ############################################################
            
            ############################################################
            # IMG LEVEL SCORE
            t_3_cpu = record_cpu()
            if self.cuda_active:
                t_3_gpu = record_gpu(t_3_gpu)

            score = self.calc_img_score(score_patches=score_patches)
            # IMG LEVEL SCORE
            ############################################################
            
            ############################################################
            # AMOMALY MAP
            t_4_cpu = record_cpu()
            if self.cuda_active:
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
            return [self.model(x)[0][:,self.idx_chosen,:,:]]
        else:
            return self.model(x)
            
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
            # pooled_features = adaptive_pooling(feature, self.pooling_strategy)#torch.nn.AvgPool2d(3, 1, 1)(feature) # TODO replace with adaptive pooling
            if type(self.pooling_strategy) == list:
                for strategy in self.pooling_strategy:
                    pooled_feature = adaptive_pooling(feature, strategy)
                    selected_features.append(pooled_feature)
            else:
                pooled_feature = adaptive_pooling(feature, self.pooling_strategy)
                selected_features.append(pooled_feature)
            ####
            # insert dim reduction here TODO 
            # after pooling
            ####
            # selected_features.append(pooled_features)
        
        concatenated_features = embedding_concat_frame(embeddings=selected_features, cuda_active=self.cuda_active)
        
        if batch_size_1:
            flattened_features = np.array(reshape_embedding(np.array(concatenated_features)))
        else:
            flattened_features = np.array([np.array(reshape_embedding(np.array(concatenated_features[k,...].unsqueeze(0)))) for k in range(batch_size)])
        
        if (self.reduce_via_std or self.reduce_via_entropy or self.reduce_via_entropy_normed) and not self.prune_output_layer[0]:
            flattened_features = np.take(flattened_features, self.idx_chosen, axis=1)#indices=#[:,self.idx_with_high_std]
        
        if self.normalize:
            flattened_features = (flattened_features - self.mean) / self.std
            
        return flattened_features
        
    def calc_score_patches(self, embeddings, batch_size_1):
        '''
        calc score_patches from which image score and anomaly map can be derived.
        '''
        if batch_size_1:
            if self.faiss:
                score_patches, _ = self.index.search(embeddings , k=args.n_neighbors)
            elif self.own_knn:
                score_patches = self.knn(torch.from_numpy(embeddings))[0].cpu().detach().numpy() # .cuda()
            else:
                score_patches, _ = self.nbrs.kneighbors(embeddings)
        else:
            if self.faiss:
                score_patches = [self.index.search(element, k=args.n_neighbors)[0] for element in embeddings]
            elif self.own_knn:
                score_patches = [self.knn(torch.from_numpy(element)[0].cpu().detach().numpy()) for element in embeddings] #.cuda()
            else:
                score_patches = [self.nbrs.kneighbors(element) for element in embeddings]
        
        return score_patches

    def calc_img_score(self, score_patches):
        '''
        calculates the image score based on score_patches
        '''
        if self.adapted_score_calc:
            score = modified_kNN_score_calc(score_patches=score_patches)
        else:
            N_b = score_patches[np.argmax(score_patches[:,0])] # only the closest val is relevant for selection! # this changes with adapted version.
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

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'/mnt/crucial/UNI/IIIT_Muen/MA/MVTechAD') #/mnt/crucial/UNI/IIIT_Muen/MA/MVTechAD\\own\\train
    parser.add_argument('--category', default='own', choices=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'])
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

if __name__ == '__main__':

    args = get_args()
    
    model = PatchCore(hparams=args)
    if args.phase == 'train':
        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=1) # allow gpu for training    
        trainer.fit(model)
        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=0) # but not for testing
        trainer.test(model)
    elif args.phase == 'test':
        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=0)
        trainer.test(model)

