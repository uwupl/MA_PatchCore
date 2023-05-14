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

def run(model, only_accuracy=False):
    '''
    Tests given config for all categories and measures inference time for own dataset.
    '''
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
            
if __name__ == '__main__':
    
    ############ DO NOT CHANGE ############
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
    ############ DO NOT CHANGE ############
    
    # RUN
    res_path = r'/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_PatchCore/results/'
    failed_runs = np.array(['None'], dtype=str)
    model.coreset_sampling_ratio = 0.01 #1%
    model.layer_cut = True
    model.faiss_standard = True
    model.own_knn = False
    
    run_counter = 0
    total_runs = 6*3*15*10
    for model_type in ['RN18', 'RN34','WRN50']:
        for layers_needed in [[1], [2], [3], [4], [1,2], [1,3], [1,4], [2,3], [2,4], [3,4], [1,2,3], [1,2,4], [1,3,4], [2,3,4], [1,2,3,4]]:
            np.save(os.path.join(res_path, f'{this_run_id}_failed_runs.npy'), failed_runs)
            model.exclude_relu = False
            model.reduce_via_entropy = True
            model.model_id = model_type
            reduction_factors = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            for factor in reduction_factors:
                model.reduction_factor = factor
                layers_str = '_'.join(str(x) for x in layers_needed)
                model.group_id = this_run_id + f'with_ReLu_{model_type}_reduced_entropy_{factor}_layers_{layers_str}'
                if not os.path.exists(os.path.join(res_path, model.group_id)):
                    try:
                        print('Run ', run_counter+1, ' of ', total_runs, ' started.')
                        run(model, True)
                        run_counter += 1
                    except:
                        failed_runs = np.append(failed_runs, model.group_id)
                        print('FAILED: ', model.group_id)
                        
            model.exclude_relu = False
            model.reduce_via_entropy = False
            model.reduce_via_entropy_normed = True
            model.model_id = model_type
            for factor in reduction_factors:
                model.reduction_factor = factor
                layers_str = '_'.join(str(x) for x in layers_needed)
                model.group_id = this_run_id + f'with_ReLu_{model_type}_reduced_entropy_normed_{factor}_layers_{layers_str}'
                if not os.path.exists(os.path.join(res_path, model.group_id)):
                    try:
                        print('Run ', run_counter+1, ' of ', total_runs, ' started.')
                        run(model, False)
                        run_counter += 1
                    except:
                        failed_runs = np.append(failed_runs, model.group_id)
                        print('FAILED: ', model.group_id)

            model.exclude_relu = False
            model.reduce_via_entropy = False
            model.reduce_via_entropy_normed = False
            model.reduce_via_std = True
            model.model_id = model_type
            for factor in reduction_factors:
                model.reduction_factor = factor
                layers_str = '_'.join(str(x) for x in layers_needed)
                model.group_id = this_run_id + f'with_ReLu_{model_type}_reduced_std_{factor}_layers_{layers_str}'
                if not os.path.exists(os.path.join(res_path, model.group_id)):
                    try:
                        print('Run ', run_counter+1, ' of ', total_runs, ' started.')
                        run(model, True)
                        run_counter += 1
                    except:
                        failed_runs = np.append(failed_runs, model.group_id)
                        print('FAILED: ', model.group_id)
                        
            model.exclude_relu = False
            model.sigmoid_in_last_layer = True
            model.reduce_via_entropy = True
            model.model_id = model_type
            # reduction_factors = [10, 20, 30, 40, 50, 60, 70, 80, 90]
            for factor in reduction_factors:
                model.reduction_factor = factor
                layers_str = '_'.join(str(x) for x in layers_needed)
                model.group_id = this_run_id + f'with_Sigmoid_{model_type}_reduced_entropy_{factor}_layers_{layers_str}'
                if not os.path.exists(os.path.join(res_path, model.group_id)):
                    try:
                        print('Run ', run_counter+1, ' of ', total_runs, ' started.')
                        run(model, True)
                        run_counter += 1
                    except:
                        failed_runs = np.append(failed_runs, model.group_id)
                        print('FAILED: ', model.group_id)
                        
            model.exclude_relu = False
            model.reduce_via_entropy = False
            model.reduce_via_entropy_normed = True
            model.model_id = model_type
            for factor in reduction_factors:
                model.reduction_factor = factor
                layers_str = '_'.join(str(x) for x in layers_needed)
                model.group_id = this_run_id + f'with_Sigmoid_{model_type}_reduced_entropy_normed_{factor}_layers_{layers_str}'
                if not os.path.exists(os.path.join(res_path, model.group_id)):
                    try:
                        print('Run ', run_counter+1, ' of ', total_runs, ' started.')
                        run(model, True)
                        run_counter += 1
                    except:
                        failed_runs = np.append(failed_runs, model.group_id)
                        print('FAILED: ', model.group_id)

            model.exclude_relu = False
            model.reduce_via_entropy = False
            model.reduce_via_entropy_normed = False
            model.reduce_via_std = True
            model.model_id = model_type
            for factor in reduction_factors:
                model.reduction_factor = factor
                layers_str = '_'.join(str(x) for x in layers_needed)
                model.group_id = this_run_id + f'with_Sigmoid_{model_type}_reduced_std_{factor}_layers_{layers_str}'
                if not os.path.exists(os.path.join(res_path, model.group_id)):
                    try:
                        print('Run ', run_counter+1, ' of ', total_runs, ' started.')
                        run(model, True)
                        run_counter += 1
                    except:
                        failed_runs = np.append(failed_runs, model.group_id)
                        print('FAILED: ', model.group_id)

    print('Failed runs: ', failed_runs)
        
    # model.exclude_relu = True
    # model.model_id = 'WRN50'
    # model.group_id = this_run_id + 'without_ReLu_WRN50'
    # if not os.path.exists(os.path.join(res_path, model.group_id)):
    #     try:
    #         print('Run ', run_counter+1, ' of ', total_runs, ' started.')
    #         run(model, True)
    #         run_counter += 1
    #     except:
    #         failed_runs = np.append(failed_runs, model.group_id)
    #         print('FAILED: ', model.group_id)
    
    # model.exclude_relu = False
    # model.sigmoid_in_last_layer = True
    # model.model_id = 'WRN50'
    # model.group_id = this_run_id + 'with_Sigmoid_WRN50'
    # if not os.path.exists(os.path.join(res_path, model.group_id)):
    #     try:
    #         print('Run ', run_counter+1, ' of ', total_runs, ' started.')
    #         run(model, True)
    #         run_counter += 1
    #     except:
    #         failed_runs = np.append(failed_runs, model.group_id)
    #         print('FAILED: ', model.group_id)
    
    # model.exclude_relu = False 
    # model.model_id = 'WRN50'
    # model.group_id = this_run_id + 'with_ReLu_WRN50'
    # if not os.path.exists(os.path.join(res_path, model.group_id)):
    #     try:
    #         print('Run ', run_counter+1, ' of ', total_runs, ' started.')
    #         run(model, True)
    #         run_counter += 1
    #     except:
    #         failed_runs = np.append(failed_runs, model.group_id)
    #         print('FAILED: ', model.group_id)
    
    # model.exclude_relu = True
    # model.model_id = 'WRN50'
    # model.group_id = this_run_id + 'without_ReLu_WRN50'
    # if not os.path.exists(os.path.join(res_path, model.group_id)):
    #     try:
    #         print('Run ', run_counter+1, ' of ', total_runs, ' started.')
    #         run(model, True)
    #         run_counter += 1
    #     except:
    #         failed_runs = np.append(failed_runs, model.group_id)
    #         print('FAILED: ', model.group_id)
            
    # model.exclude_relu = False
    # model.sigmoid_in_last_layer = True 
    # model.model_id = 'WRN50'
    # model.group_id = this_run_id + 'with_Sigmoid_WRN50'
    # if not os.path.exists(os.path.join(res_path, model.group_id)):
    #     try:
    #         print('Run ', run_counter+1, ' of ', total_runs, ' started.')
    #         run(model, True)
    #         run_counter += 1
    #     except:
    #         failed_runs = np.append(failed_runs, model.group_id)
    #         print('FAILED: ', model.group_id)
    
            