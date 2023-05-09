import numpy as np
# import numba as nb
from train_main import PatchCore#, get_args
import pytorch_lightning as pl
import os
import torch
import gc

from utils import get_summary_df

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

def run(model):
    '''
    Tests given config for all categories and measures inference time for own dataset.
    '''
    cats = ['carpet','bottle', 'cable', 'capsule', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'own']
    for cat in cats:
        model.category = cat
        print('\n\n', cat, '\n\n')
        if cat == 'own':
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
            
if __name__ == '__main__':
    # INITIALIZATION
    this_run_id = input('Please enter a run id: ')
    args = get_args()
    model = PatchCore(args=args)
    model.group_id = this_run_id + '_default_Patchcore'
    # DEFINE SETTINGS
    # feature extraction
    model.model_id = 'WRN50'
    model.layers_needed = [2,3]
    model.pooling_strategy = 'default' # nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    model.exclude_relu = False # relu won't be used for final layer, in order to not lose negative values
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
    model.n_neighbors = 5
    model.n_next_patches = 5 # only for adapted_score_calc
    # channel reduction
    model.reduce_via_std = False
    model.reduce_via_entropy = False
    model.reduce_via_entropy_normed = False
    model.reduction_factor = 50 # only for reduce_via_std or reduce_via_entropy or reduce_via_entropy_normed
    
    # RUN
    run(model)
    # WRN50
    # default
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_WRN50'
    model.layer_cut = True
    run(model)
    
    # normalizing
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_normalizing_WRN50'
    model.normalize = True
    run(model)
    
    # pooling
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_adapted_pooling_WRN50'
    model.normalize = False
    model.pooling_strategy = 'first_trial'
    run(model)
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_double_pooling_WRN50'
    model.pooling_strategy = ['first_trial', 'max_1']
    run(model)
    
    # channel reduction
    # normal pooling
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_adapted_pooling_and_red_via_entropy_50_WRN50'
    model.pooling_strategy = 'first_trial'
    model.reduce_via_entropy = True # 50
    run(model)
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_adapted_pooling_and_red_via_entropy_75_WRN50'
    model.pooling_strategy = 'first_trial'
    model.reduce_via_entropy = True 
    model.reduction_factor = 75
    run(model)
    # double pooling
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_double_pooling_and_red_via_entropy_50_WRN50'
    model.pooling_strategy = ['first_trial', 'max_1']
    model.reduce_via_entropy = True
    model.reduction_factor = 50 #
    run(model)
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_double_pooling_and_red_via_entropy_75_WRN50'
    model.pooling_strategy = ['first_trial', 'max_1']
    model.reduce_via_entropy = True
    model.reduction_factor = 75 #
    run(model)
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_double_pooling_and_red_via_entropy_87_5_WRN50'
    model.pooling_strategy = ['first_trial', 'max_1']
    model.reduce_via_entropy = True
    model.reduction_factor = 100 - 12.5 #
    run(model)
    
    # RN18
    model.group_id = this_run_id + '_default_Patchcore_RN18'
    # DEFINE SETTINGS
    # feature extraction
    model.model_id = 'RN18'
    model.layers_needed = [2,3]
    model.pooling_strategy = 'default' # nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    model.exclude_relu = False # relu won't be used for final layer, in order to not lose negative values
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
    model.n_neighbors = 5
    model.n_next_patches = 5 # only for adapted_score_calc
    # channel reduction
    model.reduce_via_std = False
    model.reduce_via_entropy = False
    model.reduce_via_entropy_normed = False
    model.reduction_factor = 50 # only for reduce_via_std or reduce_via_entropy or reduce_via_entropy_normed
    # RUN
    run(model)
    
    # default
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_RN18'
    model.layer_cut = True
    run(model)
    
    # normalizing
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_normalizing_RN18'
    model.normalize = True
    run(model)
    
    # pooling
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_adapted_pooling_RN18'
    model.normalize = False
    model.pooling_strategy = 'first_trial'
    run(model)
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_double_pooling_RN18'
    model.pooling_strategy = ['first_trial', 'max_1']
    run(model)
    
    # channel reduction
    # normal pooling
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_adapted_pooling_and_red_via_entropy_50_RN18'
    model.pooling_strategy = 'first_trial'
    model.reduce_via_entropy = True # 50
    run(model)
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_adapted_pooling_and_red_via_entropy_75_RN18'
    model.pooling_strategy = 'first_trial'
    model.reduce_via_entropy = True 
    model.reduction_factor = 75
    run(model)
    # double pooling
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_double_pooling_and_red_via_entropy_50_RN18'
    model.pooling_strategy = ['first_trial', 'max_1']
    model.reduce_via_entropy = True
    model.reduction_factor = 50 #
    run(model)
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_double_pooling_and_red_via_entropy_75_RN18'
    model.pooling_strategy = ['first_trial', 'max_1']
    model.reduce_via_entropy = True
    model.reduction_factor = 75 #
    run(model)
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_double_pooling_and_red_via_entropy_87_5_RN18'
    model.pooling_strategy = ['first_trial', 'max_1']
    model.reduce_via_entropy = True
    model.reduction_factor = 100 - 12.5 #
    run(model)
    
    # get pandas dataframe
    res_path = r'/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_PatchCore/results/'
    all_items_in_results = os.listdir(res_path)
    this_run_id = this_run_id#input('Please enter the run id: ')
    this_run_dirs = [this_dir for this_dir in all_items_in_results if this_dir.startswith(this_run_id)]
    summary_df  = get_summary_df(this_run_id)
    file_path = os.path.join(res_path, 'csv', f'summary_of_this_{this_run_id}.csv')
    summary_df.to_csv(file_path, index=False)
    
    
    