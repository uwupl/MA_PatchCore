import numpy as np
# import numba as nb
from utils.utils import remove_uncomplete_runs, remove_test_dir
from train_main import PatchCore, one_run_of_model
import pytorch_lightning as pl
import os
import torch
import sys
import traceback
import gc
import time

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
                        one_run_of_model(model)
                    else:
                        model.measure_inference = False
                        model.cuda_active_training = True
                        model.cuda_active = True
                        model.num_workers = 12
                        one_run_of_model(model)
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                et = time.perf_counter()
                print('SUCCESS\nTotal time: ', round(et-st, 2), 's')
                self.successful_runs += 1
                self.run_times = np.append(self.run_times, et-st)
            except Exception:
                ex_type, ex, tb = sys.exc_info()
                # traceback.print_tb(tb)
                traceback.print_exception(ex_type, ex, tb)
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
                        trainer = pl.Trainer(max_epochs=1, inference_mode=True, enable_model_summary=False)
                        # trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='gpu', devices=1, precision = '32') # allow gpu for training    
                        
                        trainer.fit(model)
                        model.cuda_active = False
                        # trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='cpu', devices=1, precision='32') # but not for testing
                        trainer = pl.Trainer(max_epochs=1, inference_mode=True, enable_model_summary=False)
                        trainer.test(model)    
                    else:
                        model.measure_inference = False
                        model.cuda_active_training = True
                        model.cuda_active = False
                        model.num_workers = 12
                        trainer = pl.Trainer(max_epochs=1, inference_mode=True, enable_model_summary=False)
                        # trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='gpu', devices=1, precision = '32') # allow gpu for training    
                        trainer.fit(model)
                        # trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='gpu', devices=1, precision='32') # but not for testing
                        trainer = pl.Trainer(max_epochs=1, inference_mode=True, enable_model_summary=False)
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
    model = PatchCore()
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
    
    import warnings
    warnings.filterwarnings("ignore") 

    model = get_default_PatchCoreModel()
    manager = TestContainer()
    manager.total_runs = 2*4*3
    model.adapted_score_calc = True
    model.n_neighbors = 4
    model.n_next_patches = 16
    model.layer_cut = True
    run_id_prefix = '0207_quantized_qint8_Layer_Comp-'
    for model_id in ['RN34', 'RN50', 'RN18', 'WRN50']:# 'WRN101', 'WRN152']:
        model.model_id = model_id
        for layers_needed in [[1], [2], [3]]:
            if layers_needed == [1]:
                model.coreset_sampling_method = 'random_selection' # whole dataset does not fit on the gpu and cpu is too slow
                model.specific_number_of_examples = 1000*4
            elif layers_needed == [2]:
                model.coreset_sampling_method = 'k_center_greedy'
                model.specific_number_of_examples = 1000*2
            elif layers_needed == [3]:
                model.coreset_sampling_method = 'k_center_greedy'
                model.specific_number_of_examples = 1000*1 # in order to get the same amount of data points for each layer
            model.layers_needed = layers_needed
            model.quantize_qint8 = True
            model.group_id = f'{run_id_prefix}-quantized-Layer {layers_needed[0]} of {model.model_id}'
            manager.this_run_id = run_id_prefix
            manager.run(model)
            
            model.quantize_qint8 = False
            model.group_id = f'{run_id_prefix}-not_quantized-Layer {layers_needed[0]} of {model.model_id}'
            manager.this_run_id = run_id_prefix
            manager.run(model)
    
    print(manager.get_summarization())
