import numpy as np
# import numba as nb
from utils import remove_uncomplete_runs, remove_test_dir
from train_main import PatchCore
import pytorch_lightning as pl
import os
import torch
import sys
import traceback
import gc
import time

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

class TestContainer():
    '''
    Class which handles a test run.
    '''
    def __init__(self) -> None:
        self.run_no = 0
        self.this_run_id = ''
        self.failed_runs = np.array(['None'], dtype=str)
        self.failed_runs_no = 0
        self.dir_exists = 0
        self.successful_runs = 0
        self.total_runs = 100 #TODO
        self.res_path = r'/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_PatchCore/results/'
        self.run_times = np.array([])
        
    def run(self, model, only_accuracy=False):#, res_path = r'/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_PatchCore/results/'):
        '''
        Tests given config for all categories and measures inference time for own dataset.
        '''
        if not os.path.exists(os.path.join(self.res_path, model.group_id)):
            try:
                print('Run ', self.run_no+1, ' of ', self.total_runs, ' started.')
                st = time.perf_counter()
                cats = ['carpet','bottle','own', 'cable', 'capsule', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
                for cat in cats:
                    model.category = cat
                    print('\n\n', cat, '\n\n')
                    if cat == 'own' and not only_accuracy:
                        model.measure_inference = True
                        model.cuda_active_training = True
                        model.cuda_active = False
                        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='gpu', devices=1, precision = '32') # allow gpu for training    
                        trainer.fit(model)
                        model.cuda_active = False
                        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='cpu', devices=1, precision='32') # but not for testing
                        trainer.test(model)    
                    else:
                        model.measure_inference = False
                        model.cuda_active_training = True
                        model.cuda_active = True
                        model.num_workers = 12
                        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='gpu', devices=1, precision = '32') # allow gpu for training    
                        trainer.fit(model)
                        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='cpu', devices=1, precision='32') # but not for testing
                        trainer.test(model)    
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                et = time.perf_counter()
                print('SUCCESS\nTotal time: ', round(et-st, 2), 's')
                self.successful_runs += 1
                self.run_times = np.append(self.run_times, et-st)
            except Exception:
                ex_type, ex, tb = sys.exc_info()
                traceback.print_tb(tb)
                self.failed_runs = np.append(self.failed_runs, model.group_id)
                self.failed_runs_no += 1
                np.save(os.path.join(self.res_path, f'{self.this_run_id}_failed_runs.npy'), self.failed_runs)
                print('FAILED: ', model.group_id)
        else:
            self.dir_exists += 1
            print('Directory already exists: ', model.group_id)
        self.run_no += 1

    def run_on_pi(self, model, only_accuracy=False):#, res_path = r'/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_PatchCore/results/'):
        '''
        Tests given config for all categories and measures inference time for own dataset.
        '''
        if not os.path.exists(os.path.join(self.res_path, model.group_id)):
            try:
                print('Run ', self.run_no+1, ' of ', self.total_runs, ' started.')
                st = time.perf_counter()
                cats = ['own','carpet','bottle', 'cable', 'capsule', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
                for cat in cats:
                    model.category = cat
                    print('\n\n', cat, '\n\n')
                    if cat == 'own' and not only_accuracy:
                        model.measure_inference = True
                        model.cuda_active_training = True
                        model.cuda_active = True
                        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='gpu', devices=1, precision = '32') # allow gpu for training    
                        trainer.fit(model)
                        model.cuda_active = False
                        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='cpu', devices=1, precision='32') # but not for testing
                        trainer.test(model)    
                    else:
                        model.measure_inference = False
                        model.cuda_active_training = True
                        model.cuda_active = False
                        model.num_workers = 12
                        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='gpu', devices=1, precision = '32') # allow gpu for training    
                        trainer.fit(model)
                        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='gpu', devices=1, precision='32') # but not for testing
                        trainer.test(model)    
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                et = time.perf_counter()
                print('SUCCESS\nTotal time: ', round(et-st, 2), 's')
                self.successful_runs += 1
                self.run_times = np.append(self.run_times, et-st)
            except Exception:
                ex_type, ex, tb = sys.exc_info()
                traceback.print_tb(tb)
                self.failed_runs = np.append(self.failed_runs, model.group_id)
                self.failed_runs_no += 1
                np.save(os.path.join(self.res_path, f'{self.this_run_id}_failed_runs.npy'), self.failed_runs)
                print('FAILED: ', model.group_id)
        else:
            self.dir_exists += 1
            print('Directory already exists: ', model.group_id)
        self.run_no += 1
    
    def get_summarization(self):
        '''
        Returns a summarization of the test run.
        '''
        try:
            remove_uncomplete_runs()
            remove_test_dir()
        except:
            print('Could not remove test directory or were unable to remove uncomplete runs.')
        return f'Run {self.this_run_id} finished.\n{self.successful_runs} of {self.total_runs} runs were successful.\n{self.failed_runs_no} runs failed.\n{self.dir_exists} directories already existed and were skipped.\nAverage time per run: {np.mean(self.run_times)}s.\nStandard deviation: {np.std(self.run_times)}s.\nMedian: {np.median(self.run_times)}s.\nTotal Time: {np.sum(self.run_times)}s.'#\nMaximum: {np.max(self.run_times)}s.'

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
            
if __name__ == '__main__':

    args = get_args()
    model = get_default_PatchCoreModel()#args=args)
    manager = TestContainer()
    model.model_id = 'RN34'
    model.layers_needed = [2]
    model.adapted_score_calc = True
    model.n_neighbors = 4
    model.n_next_patches = 16
    # model.reduce_via_std = True
    model.layer_cut = True
    # model.prune_output_layer = (True, [])
    # model.reduction_factor = 30
    

    run_id_prefix = 'prune_iterative_2306-'
    # pruning_factors = [0.4,0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    # pruning_factors = [0.03505]

    # for pf in pruning_factors:
    #     model.prune_l1_unstructured = (True, pf)
    #     model.group_id = run_id_prefix + 'pruned_by_' + str(pf)
    #     manager.run(model)
    # model.prune_l1_unstructured = (False, 0.0)    
    # model.group_id = run_id_prefix + 'non_pruned'
    # manager.run(model)
    # manager.get_summarization()

    # model.reduce_via_std = True
    # model.reduce_via_entropy_normed = True
    model.faiss_standard = True
    model.own_knn = False
    # model.sigmoid_in_last_layer = True

    # model.reduction_factor = 80
    # model.prune_output_layer = (False, [])
    # model.reduction_factor = 90
    # model.reduce_via_std = True
    # model.pretrain_for_channel_selection = True
    # model.iterative_pruning = (True, 10)

    pruning_methods = ['L2']#, 'L1', 'FPGM']
    pruning_factors = [0.03] # 3% for each run
    times_list = [5,1,2,3,4,6,7,8,9,10,11,12,13,14,15]
    model.prune_output_layer = (True, [])
    model.reduce_via_std = True
    model.reduction_factor = 98
    model.iterative_pruning = (True, 0)
    model.pretrain_for_channel_selection = True
    for times in times_list:
        for pm in pruning_methods:
            for pf in pruning_factors:
                model.iterative_pruning = (True, times)
                model.sparsity = pf
                model.prune_structured_nni = (True, list(), pm) # TODO
                # model.group_id = run_id_prefix + f'iterative_pruning_(10)-pruned_by_0.4-L2-reduced_via_entropy_by_90'
                model.group_id = run_id_prefix + f'pruned_{times}_times-{pm}'
                model.pretrain_for_channel_selection = True # because this is set to False in the train_main after every run # TDOO
                manager.run(model)
    # model.prune_structured_nni = (True, 0.3, 'L2')
    # model.pretrain_for_channel_selection = False   
    # model.group_id = run_id_prefix + 'no_iterative_pruning-pruned_by_0.4-L2-reduced_via_entropy_by_85'
    # model.reduction_factor = 85
    manager.run(model)
    print(manager.get_summarization())