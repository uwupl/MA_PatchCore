import numpy as np
import numba as nb
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
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
    
    # delete outliers
    sum_of_each_patch = np.sum(score_patches,axis=1)
    threshold_val = 50*np.percentile(sum_of_each_patch, 50)
    non_outlier_patches = np.argwhere(sum_of_each_patch < threshold_val).flatten()#[0]
    if len(non_outlier_patches) < score_patches.shape[0]:
        score_patches = score_patches[non_outlier_patches]
        print('deleted outliers: ', sum_of_each_patch.shape[0]-len(non_outlier_patches))
    # score_patches = score_patches[np.sum(score_patches,axis=1) < 1e9] 
    k = score_patches.shape[1]
    
    # score_patches[score_patches >= 1e20] = 1e20
    weights = np.array([(k-i)**2 for i in range(k)])#np.divide(np.array([(k-i)**2 for i in range(k)]), 1, dtype=np.float64) # Summe(i²) = (k*(k+1)*(2*k+1))/6
    dists = np.sum(np.multiply(score_patches, weights), axis=1, dtype=np.float64)
    sorted_args = np.argsort(dists)
    score = np.zeros(n_next_patches)
    for p in range(1,n_next_patches+1):    
        N_b = score_patches[sorted_args[-p]].astype(np.float128)
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

def get_summary_df(this_run_dirs: list, res_path: str):
    '''
    Takes a list of directories and returns a dataframe with the summary of all runs
    '''
    img_auc_total = np.array([])
    img_auc_total_mean = np.array([])
    img_auc_MVTechAD_total = np.array([])
    img_auc_own = np.array([])
    backbone_storage_total = np.array([])
    backbone_flops_total = np.array([])
    feature_extraction_total = np.array([])
    embedding_of_feature_total = np.array([])   
    calc_distances_total = np.array([]) 
    calc_scores_total = np.array([])    
    total_time_total = np.array([])
    for k, run_dir in enumerate(this_run_dirs):
        file_name = 'summary_' + run_dir + '.csv'
        file_path = os.path.join(res_path, run_dir,'csv',file_name)
        pd_summary = pd.read_csv(file_path, index_col=0)
        img_auc = np.float32(pd_summary.loc['img_auc_[%]'].values)
        img_auc_mean = np.mean(img_auc)
        img_auc_own = img_auc[-1]
        img_auc_MVTechAD = np.mean(img_auc[:-1])
        backbone_storage = np.max(np.float32(pd_summary.loc['backbone_storage_[MB]'].values))
        backbone_flops = np.max(np.float32(pd_summary.loc['backbone_mult_adds_[M]'].values))
        feature_extraction = np.max(np.float32(pd_summary.loc['feature_extraction_[ms]'].values))
        embedding_of_feature = np.max(np.float32(pd_summary.loc['embedding_of_features_[ms]'].values))
        calc_distances = np.max(np.float32(pd_summary.loc['calc_distances_[ms]'].values))
        calc_scores = np.max(np.float32(pd_summary.loc['calc_scores_[ms]'].values))
        total_time = np.max(np.float32(pd_summary.loc['total_time_[ms]'].values))
        if k == 0:
            # img_auc_total = img_auc
            img_auc_total_mean = img_auc_mean
            img_auc_total_own = img_auc_own
            img_auc_MVTechAD_total = img_auc_MVTechAD
            backbone_storage_total = backbone_storage
            backbone_flops_total = backbone_flops  
            feature_extraction_total = feature_extraction
            embedding_of_feature_total = embedding_of_feature
            calc_distances_total = calc_distances
            calc_scores_total = calc_scores
            total_time_total = total_time
        else:
            # img_auc_total = np.vstack((img_auc_total, img_auc))
            img_auc_total_mean = np.vstack((img_auc_total_mean, img_auc_mean))
            img_auc_MVTechAD_total = np.vstack((img_auc_MVTechAD_total, img_auc_MVTechAD))
            img_auc_total_own = np.vstack((img_auc_total_own, img_auc_own))
            backbone_storage_total = np.vstack((backbone_storage_total, backbone_storage))
            backbone_flops_total = np.vstack((backbone_flops_total, backbone_flops))
            feature_extraction_total = np.vstack((feature_extraction_total, feature_extraction))
            embedding_of_feature_total = np.vstack((embedding_of_feature_total, embedding_of_feature))
            calc_distances_total = np.vstack((calc_distances_total, calc_distances))
            calc_scores_total = np.vstack((calc_scores_total, calc_scores))
            total_time_total = np.vstack((total_time_total, total_time))

    summary_np = np.zeros((10, len(img_auc_total_mean)))
    helper_list = [img_auc_total_mean, img_auc_MVTechAD_total, img_auc_total_own, backbone_storage_total, backbone_flops_total, feature_extraction_total, embedding_of_feature_total, calc_distances_total, calc_scores_total, total_time]
    for i, entry in enumerate(helper_list):
        summary_np[i, :] = entry.flatten()
    run_summary_dict = {}
    for k in range(len(img_auc_total_mean)):
        print(k)
        for a, b in zip(summary_np[:,k].flatten(), ['img_auc_mean', 'img_auc_MVTechAD', 'img_auc_own','backbone_storage', 'backbone_flops', 'feature_extraction', 'embedding_of_feature', 'calc_distances', 'calc_scores', 'total_time']):
            if k == 0:
                run_summary_dict[b] = [float(a)]
            else:
                run_summary_dict[b] += [float(a)]

    index_list = [name[len(name.split('_')[0])+1:] for name in this_run_dirs]
    run_summary_df = pd.DataFrame(run_summary_dict, index=index_list)
    return run_summary_df

def plot_results(labels, feature_extraction, embedding, search, calc_scores, own_auc, MVTechAD_auc, storage): #TODO storage
    '''
    visualizes results in bar chart
    '''
    for k in range(len(labels)):
        labels[k] = labels[k] + '\n' + str(storage[k])
    
    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots(figsize=(20,10))
    ax_2 = ax.twinx()
    rects1 = ax.bar(x - 0.5*width, feature_extraction, width, label='feature extraction', color = 'crimson')
    rects2 = ax.bar(x - 0.5*width, embedding, width, label='embedding', bottom=feature_extraction, color = 'purple')
    rects3 = ax.bar(x - 0.5*width, search, width, label='search', bottom=list(np.array(embedding) + np.array(feature_extraction)), color = 'slateblue')
    rects4 = ax.bar(x - 0.5*width, calc_scores, width, label='calc scores',bottom=list(np.array(embedding) + np.array(feature_extraction) + np.array(search)), color = 'darkgoldenrod')
    # rects4 = ax.bar(x - 0.5*width, anomaly_map, width, label='anomaly map',bottom=list(np.array(embedding_cpu) + np.array(feature_extraction_cpu) + np.array(search_memory)), color = 'darkgoldenrod')
    rects_1 = ax_2.bar(x + 0.25 * width, own_auc, width*0.3, label = 'Own Auc', color = 'black')
    rects_2 = ax_2.bar(x + 0.75 * width, MVTechAD_auc, width*0.3, label = 'MVTechAD Auc', color = 'grey')
    # rects5 = ax.bar(x + width, total_cpu, width, label='total')
    # rects3 = ax.bar(x, )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('elapsed time per sample [ms] (mean)')
    ax.set_title('Process time for different feautures maps')
    ax_2.set_ylabel('Auccarcy')
    ax_2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    ax.set_xticks(x, labels)
    ax.legend()
    ax_2.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    # ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3, fmt='%1.3f')
    ax_2.bar_label(rects_1, padding=3,fmt='%1.1f')
    ax_2.bar_label(rects_2, padding=3,fmt='%1.1f')
    ax_2.set_yticks([20,40,60,80,100])
    ax_2.set

    fig.tight_layout()

    # plt.savefig(os.path.join(plot_dir, '13_adapt_max_pool.svg'), bbox_inches = 'tight')

    plt.show()

