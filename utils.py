import numpy as np
import numba as nb
import torch
import cv2
import os
import warnings

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min) 

def record_gpu(cuda_event):
    '''
    gpu_measurement
    '''
    cuda_event.record()
    torch.cuda.synchronize()
    
    return cuda_event

def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)
    return dist

@nb.jit(nopython=True)
def modified_kNN_score_calc_old(score_patches):
    k = score_patches.shape[1]
    weights = np.divide(np.array([k-i for i in range(k)]), 1)#((k-1)*k)/2)
    # weights = np.ones(k)
    dists = np.sum(np.multiply(score_patches, weights), axis=1)
    N_b = score_patches[np.argmax(dists)]
    w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
    score = w*np.max(dists)
    return score

# @nb.jit(nopython=True)
def modified_kNN_score_calc(score_patches, n_next_patches = 5):
    
    # weights = np.divide(np.array([(k-i) for i in range(k)]), ((k-1)*k)/2)
    # weights = np.ones(k)
    # weights = np.zeros(k)
    # weights[0] = 1
    
    score_patches = score_patches[np.sum(score_patches,axis=1) < 1e34] 
    k = score_patches.shape[1]
    
    # score_patches[score_patches >= 1e20] = 1e20
    weights = np.array([(k-i)**2 for i in range(k)])#np.divide(np.array([(k-i)**2 for i in range(k)]), 1, dtype=np.float64) # Summe(i²) = (k*(k+1)*(2*k+1))/6
    dists = np.sum(np.multiply(score_patches, weights), axis=1, dtype=np.float64)
    sorted_args = np.argsort(dists)
    score = np.zeros(n_next_patches)
    for p in range(1,n_next_patches+1):    
        N_b = score_patches[sorted_args[-p]]
        exp_N_b = np.exp(N_b)
        # exp_N_b[exp_N_b >= 1e25] = 1e25
        exp_N_b_sum = np.sum(exp_N_b)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                softmax = np.divide(np.max(exp_N_b), exp_N_b_sum)
            except:
                softmax = 1.0
        w = np.float64(1.0 - softmax)
        score[p-1] =  w*dists[sorted_args[-p]]
    return np.mean(score)

def prep_dirs(root, category):
    # make embeddings dir
    embeddings_path = os.path.join('./', 'embeddings', category)
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    return embeddings_path, sample_path, source_code_save_path