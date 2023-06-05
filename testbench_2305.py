import numpy as np
# import numba as nb
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
                        model.cuda_active = True
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
    # print('sleeep...')
    # time.sleep(400 * 35)
    # print('awake!')
    this_run_id = '0406_4_'
    args = get_args()
    model = get_default_PatchCoreModel()#args=args)
    manager = TestContainer()
    manager.this_run_id = this_run_id

    model.model_id = 'RN34'
    model.n_neighbors = 20
    model.layers_needed = [2]
    model.layer_cut = True
    model.sigmoid_in_last_layer = True
    model.reduce_via_entropy_normed = True
    model.normalize = False
    model.own_knn = False
    model.faiss_standard = True
    # model.prune_output_layer = (True, [])
    pooling_strategies = ['avg311']#[['avg522'], ['avg311'], ['avg311', 'max311']] # ['avg522', 'max522'],
    reduction_factors = [95]#[50,80,90,95,100]
    # pooling_strategies = ['avg110', 'avg311', 'avg321', 'avg331', 'avg512', 'avg522', 'avg532', 'avg713', 'avg723', 'avg733', 'avg914', 'avg924', 'avg934', 'max110', 'max311', 'max321', 'max331', 'max512', 'max522', 'max532', 'max713', 'max723', 'max733', 'max914', 'max924', 'max934']# 'first_trial', 'second_trial', 'max_1']
    # n_neighbors = [1,3,9,18,36,72,144]
    # n_next_patches = [1,3,5,10,20]
    n_neighbors = [1,2,3,4,5,7]
    n_next_patches = [10, 12, 14, 16, 18, 20]
    manager.total_runs = len(pooling_strategies) * len(reduction_factors) * len(n_neighbors) * (len(n_next_patches) + 1)
    model.adapted_score_calc = True
    for n_n in n_neighbors:
        model.n_neighbors = n_n
        for n_np in n_next_patches:
            model.n_next_patches = n_np
            for reduction_factor in reduction_factors:
                if reduction_factor == 100:
                    model.reduce_via_entropy_normed = False
                else:
                    model.reduce_via_entropy_normed = True
                model.reduction_factor = reduction_factor
                for pooling_strategy in pooling_strategies:
                    model.pooling_strategy = pooling_strategy
                    model.group_id = this_run_id + str(pooling_strategy) + '-reduced_by_' + str(reduction_factor) + '-RN34-L_2_normalized-entropy_normed-' + str(n_n) + '_NN_' + str(n_np) + 'NP (adapted)'
                    manager.run(model, True)
    model.adapted_score_calc = False
    for n_n in n_neighbors:
        model.n_neighbors = n_n
        for reduction_factor in reduction_factors:
            if reduction_factor == 100:
                model.reduce_via_entropy_normed = False
            else:
                model.reduce_via_entropy_normed = True
            model.reduction_factor = reduction_factor
            for pooling_strategy in pooling_strategies:
                model.pooling_strategy = pooling_strategy
                model.group_id = this_run_id + str(pooling_strategy) + '-reduced_by_' + str(reduction_factor) + '-RN34-L_2_normalized-entropy_normed-' + str(n_n) + '_NN_' + '(not adapted)'
                manager.run(model, True)               
    # model.reduce_via_entropy_normed = False
    # model.reduce_via_std = True
    # for reduction_factor in reduction_factors:
    #     if reduction_factor == 100:
    #         model.reduce_via_std = False
    #     else:
    #         model.reduce_via_std = True
    #     model.reduction_factor = reduction_factor
    #     for pooling_strategy in pooling_strategies:
    #         model.pooling_strategy = pooling_strategy
    #         model.group_id = this_run_id + '-' + str(pooling_strategy) + '-reduced_' + str(reduction_factor) + '-RN34-layer_2-std'
    #         run(model, False)
    # for pooling_strategy in pooling_strategies:
    #     model.pooling_strategy = pooling_strategy
    #     model.group_id = this_run_id + '-' + str(pooling_strategy) + '-RN34-layer_2'
    #     run(model, False)
    print(manager.get_summarization())