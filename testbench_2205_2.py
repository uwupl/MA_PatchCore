import numpy as np
# import numba as nb
from train_main import PatchCore#, get_args
import pytorch_lightning as pl
import os
import torch
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

def run(model, only_accuracy=False):
    '''
    Tests given config for all categories and measures inference time for own dataset.
    '''
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
    print('Total time: ', round(et-st, 2), 's')
            
def get_default_PatchCoreModel(args):
    '''
    Returns a PatchCore model with default settings.
    '''
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
    
    this_run_id = 'pooling_test_2205_2'
    args = get_args()
    model = get_default_PatchCoreModel(args=args)

    model.model_id = 'RN34'
    model.n_neighbors = 20
    model.layers_needed = [2]
    model.layer_cut = True
    model.sigmoid_in_last_layer = True
    model.reduce_via_entropy_normed = True
    model.normalize = True
    pooling_strategies = [['avg522'], ['avg311'], ['avg311', 'max311']] # ['avg522', 'max522'],
    reduction_factors = [5,10,20,50,80,90,95,100]
    # pooling_strategies = ['avg110', 'avg311', 'avg321', 'avg331', 'avg512', 'avg522', 'avg532', 'avg713', 'avg723', 'avg733', 'avg914', 'avg924', 'avg934', 'max110', 'max311', 'max321', 'max331', 'max512', 'max522', 'max532', 'max713', 'max723', 'max733', 'max914', 'max924', 'max934']# 'first_trial', 'second_trial', 'max_1']
    for reduction_factor in reduction_factors:
        if reduction_factor == 100:
            model.reduce_via_entropy_normed = False
        else:
            model.reduce_via_entropy_normed = True
        model.reduction_factor = reduction_factor
        for pooling_strategy in pooling_strategies:
            model.pooling_strategy = pooling_strategy
            model.group_id = this_run_id + '-' + str(pooling_strategy) + '-reduced_' + str(reduction_factor) + '-RN34-L_2_normalized-entropy_normed'
            run(model, False)
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