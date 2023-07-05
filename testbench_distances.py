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
    model.model_id = 'RN34'
    model.layers_needed = [2]
    model.adapted_score_calc = False
    # model.n_neighbors = 4
    # model.n_next_patches = 16
    model.layer_cut = True
    run_id_prefix = 'compare_distances-with_reduction_by_std_with_50_percent'
    manager.this_run_id = run_id_prefix
    manager.total_runs = len(model.metrices) * 2
    model.cuda_active_training = True
    model.cuda_active = True
    model.need_for_own_last_layer = True
    
    # default run
    # manager.this_run_id = run_id_prefix + 'default'
    # manager.run(model)
    
    ##############################
    # define test loop here
    metrices = { 
            0:'euclidean', # 0.88
            1:'minkowski', # nur mit p spannend
            2:'cityblock', # manhattan
            3:'chebyshev',
            4:'cosine',
            5:'correlation',
            6:'hamming',
            7:'jaccard',
            8:'braycurtis',
            9:'canberra',
            10:'jensenshannon',
            # 11:'matching', # sysnonym for hamming
            11:'dice',
            12:'kulczynski1',
            13:'rogerstanimoto',
            14:'russellrao',
            15:'sokalmichener',
            16:'sokalsneath',
            # 18:'wminkowski',
            17:'mahalanobis',
            18:'seuclidean',
            19:'sqeuclidean',
            }
    model.reduce_via_std = True
    model.reduction_factor = 50
    model.prune_output_layer = (True, [])
    
    metrics_that_need_p = [1]#, 18]
    p_vals = [1, 2, 3, 4, 5]
    feasible_distances = ['minkowski', 'cityblock', 'canberra', 'braycurtis', 'sqeuclidean', 'jensenshannon', 'chebyshev', 'mahalanobis', 'cosine', 'correlation']
    
    for i in range(len(metrices)):
        if metrices[i] in feasible_distances:
            model.adapted_score_calc = False
            model.metric_id = i
            if i in metrics_that_need_p:
                for p in p_vals:
                    model.group_id = f'{run_id_prefix}-{metrices[i]}_p={p}-non_adapted_score_calc'
                    model.metrics_p = p
                    manager.run(model, True)
            else:            
                model.metrics_p = None
                model.group_id = run_id_prefix + '-' + metrices[i] + '-' + 'non_adapted_score_calc'
                manager.run(model, True)
            
            model.adapted_score_calc = True
            model.n_neighbors = 4
            model.n_next_patches = 16
            
            if i in metrics_that_need_p:
                for p in p_vals:
                    model.group_id = f'{run_id_prefix}-{metrices[i]}_p={p}-adapted_score_calc'
                    model.metrics_p = p
                    manager.run(model, True)
            else:
                model.metrics_p = None
                model.group_id = run_id_prefix + '-' + metrices[i] + '-' + 'adapted_score_calc'
                manager.run(model, True)
    ##############################
    
    print(manager.get_summarization())
