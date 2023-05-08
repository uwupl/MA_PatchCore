import numpy as np
# import numba as nb
from train_main import PatchCore#, get_args
import pytorch_lightning as pl
import os
import torch
import gc

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
    Tests given config for all categories and measures inference time for 
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
    # run(model)
    
    # model.group_id = this_run_id + '_default_Patchcore_with_layer_cut'
    # model.layer_cut = True
    # run(model)
    
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_normalizing'
    model.normalize = True
    run(model)
    
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_adapted_pooling'
    model.normalize = False
    model.pooling_strategy = 'first_trial'
    run(model)
    
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_double_pooling'
    model.pooling_strategy = ['first_trial', 'max_1']
    run(model)
    
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_adapted_pooling_and_red_via_entropy'
    model.pooling_strategy = 'first_trial'
    model.reduce_via_entropy = True
    run(model)
    
    model.group_id = this_run_id + '_default_Patchcore_with_layer_cut_and_double_pooling_and_red_via_entropy'
    model.pooling_strategy = ['first_trial', 'max_1']
    model.reduce_via_entropy = True
    model.reduction_factor = 75
    run(model)
      
    
    