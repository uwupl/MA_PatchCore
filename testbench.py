import numpy as np
# import numba as nb
from train_main import PatchCore#, get_args
import pytorch_lightning as pl
import os

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r"/mnt/crucial/UNI/IIIT_Muen/MA/MVTechAD") #C:\Users\uwupl\IIIT Muen\MA\productive\mvtec_anomaly_detection
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
    ##################################################################    
    model = PatchCore(args=args)
    model.group_id = 'layer_cut_adapted_pooling_RN18_2'
    model.pooling_strategy = 'first_trial'
    cats = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'own']
    
    model = PatchCore(args=args)
    model.group_id = 'layer_cut_adapted_pooling_only_2nd_layer_normalized_features_reduce_via_std_50_RN18_2'
    model.pooling_strategy = 'first_trial'
    model.layers_needed = [2]
    model.normalize = True
    model.reduce_via_std = True
    model.reduction_factor = 50
    for k, cat in enumerate(cats):
            model.category = cat
            if cat == 'own':
                model.measure_inference = True
                trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='gpu', devices=1, precision = '32') # allow gpu for training    
                trainer.fit(model)
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
    
    for k, cat in enumerate(cats):
        model.category = cat
        if cat == 'own':
            model.measure_inference = True
            trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='gpu', devices=1, precision = '32') # allow gpu for training    
            trainer.fit(model)
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
    ##################################################################
    model = PatchCore(args=args)
    model.group_id = 'layer_cut_adapted_pooling_only_2nd_layer_RN18_2'
    model.pooling_strategy = 'first_trial'
    model.layers_needed = [2]
    for k, cat in enumerate(cats):
        model.category = cat
        if cat == 'own':
            model.measure_inference = True
            trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='gpu', devices=1, precision = '32') # allow gpu for training    
            trainer.fit(model)
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
    ##################################################################
    model = PatchCore(args=args)
    model.group_id = 'layer_cut_adapted_pooling_only_2nd_layer_normalized_features_RN18_2'
    model.pooling_strategy = 'first_trial'
    model.layers_needed = [2]
    model.normalize = True
    for k, cat in enumerate(cats):
        model.category = cat
        if cat == 'own':
            model.measure_inference = True
            trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='gpu', devices=1, precision = '32') # allow gpu for training    
            trainer.fit(model)
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
    ##################################################################
    model = PatchCore(args=args)
    model.group_id = 'layer_cut_adapted_pooling_only_2nd_layer_normalized_features_reduce_via_std_50_RN18_2'
    model.pooling_strategy = 'first_trial'
    model.layers_needed = [2]
    model.normalize = True
    model.reduce_via_std = True
    model.reduction_factor = 50
    for k, cat in enumerate(cats):
            model.category = cat
            if cat == 'own':
                model.measure_inference = True
                trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, accelerator='gpu', devices=1, precision = '32') # allow gpu for training    
                trainer.fit(model)
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