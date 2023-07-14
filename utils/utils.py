import numpy as np
import numba as nb
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# from train_main import PatchCore
# import pytorch_lightning as pl
import torch
import cv2
import os
import warnings
import time
# import gc
import shutil

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
    sum_of_each_patch = np.sum(score_patches,axis=1)
    threshold_val = 50*np.percentile(sum_of_each_patch, 50)
    non_outlier_patches = np.argwhere(sum_of_each_patch < threshold_val).flatten()#[0]
    if len(non_outlier_patches) < score_patches.shape[0]:
        score_patches = score_patches[non_outlier_patches]
        print('deleted outliers: ', sum_of_each_patch.shape[0]-len(non_outlier_patches))
    k = score_patches.shape[1]
    # weights = np.array([(k-i)**2 for i in range(k)])#np.divide(np.array([(k-i)**2 for i in range(k)]), 1, dtype=np.float64) # Summe(i²) = (k*(k+1)*(2*k+1))/6
    weights = np.ones(k)    
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

@nb.jit(nopython=True)
def modified_kNN_score_calc_numba(score_patches, n_next_patches = 5, outlier_deletion = True, outlier_factor = 50):
    '''
    numba version of adapted score calculation
    '''
    if outlier_deletion:
        sum_of_each_patch = np.sum(score_patches,axis=1)
        threshold_val = outlier_factor*np.percentile(sum_of_each_patch, 50)
        non_outlier_patches = np.argwhere(sum_of_each_patch < threshold_val).flatten()#[0]
        if len(non_outlier_patches) < score_patches.shape[0]:
            score_patches = score_patches[non_outlier_patches]
            print('deleted outliers: ', sum_of_each_patch.shape[0]-len(non_outlier_patches))
    k = score_patches.shape[1]
    weights = np.array([(k-i)**2 for i in range(k)])#np.divide(np.array([(k-i)**2 for i in range(k)]), 1, dtype=np.float64) # Summe(i²) = (k*(k+1)*(2*k+1))/6
    dists = np.sum(np.multiply(score_patches, weights), axis=1, dtype=np.float64)
    sorted_args = np.argsort(dists)
    score = np.zeros(n_next_patches)
    for p in range(1,n_next_patches+1):    
        N_b = score_patches[sorted_args[-p]].astype(np.float64)
        exp_N_b = np.exp(N_b)
        # exp_N_b[exp_N_b >= 1e25] = 1e25
        exp_N_b_sum = np.sum(exp_N_b)
        # with warnings.catch_warnings():
        #     warnings.filterwarnings('error')
        #     try:
        softmax = np.divide(np.max(exp_N_b), exp_N_b_sum)
            # except:
                # softmax = 1.0
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

def get_summary_df(this_run_id: str, res_path: str, save_df = False):
    '''
    Takes a run_id, reads all files and returns a dataframe with the summary of all runs
    '''
    failed_runs = []
    correction_number = 0
    
    all_items_in_results = os.listdir(res_path)
    this_run_dirs = [this_dir for this_dir in all_items_in_results if this_dir.startswith(this_run_id)]
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
        try:
            pd_summary = pd.read_csv(file_path, index_col=0)
            if pd_summary.shape[1] != int(16):
                # print(k)
                print(file_path)
                failed_runs.append(k)
                correction_number += 1
                continue
        except:
            print('file not found: ', file_path)
            failed_runs.append(k)
            correction_number += 1
            continue
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
        # coreset_size = np.max(np.float32(pd_summary.loc['coreset_size'].values))
        if (k - correction_number) == 0:
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
        # print(k)
        for a, b in zip(summary_np[:,k].flatten(), ['img_auc_mean', 'img_auc_MVTechAD', 'img_auc_own','backbone_storage', 'backbone_flops', 'feature_extraction', 'embedding_of_feature', 'calc_distances', 'calc_scores', 'total_time']):
            if k == 0:
                run_summary_dict[b] = [float(a)]
            else:
                run_summary_dict[b] += [float(a)]

    index_list = [name[len(this_run_id):] for k, name in enumerate(this_run_dirs) if k not in failed_runs]
    run_summary_df = pd.DataFrame(run_summary_dict, index=index_list)
    if save_df:
        file_path = os.path.join(res_path, 'csv', f'{int(time.time())}_summary_of_this_{this_run_id}.csv')
        run_summary_df.to_csv(file_path, index=False)
    return run_summary_df

def plot_results(labels, feature_extraction, embedding, search, calc_scores, own_auc, MVTechAD_auc, storage, fig_size = (20,10), title = 'Comparison', only_auc = False, width = 0.4, save_fig = False, res_path = r'/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_PatchCore/results/plots/', show = True):
    '''
    visualizes results in bar chart
    '''
    for k in range(len(labels)):
        # labels[k] = labels[k]
        labels[k] = labels[k] + '\n' + str(round(storage[k],2)) + ' MB'
    
    x = np.arange(len(labels))  # the label locations
    width = width  # the width of the bars
    ### temp ###
    #Direct input 
    # plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
    # #Options
    # params = {'text.usetex' : True,
    #         'font.size' : 11,
    #         'font.family' : 'lmodern',
    #         'text.latex.unicode': True,
    #         }
    # plt.rcParams.update(params)     
    ### temp ###
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)
    if not only_auc:
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
        ax.set_title(title)
        ax_2.set_ylabel('Auccarcy')
        ax_2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        ax.set_xticks(x, labels)
        ax.legend()
        ax_2.legend()

        ax.bar_label(rects1, padding=3)
        # ax.bar_label(rects2, padding=3)
        # ax.bar_label(rects3, padding=3)
        ax.bar_label(rects4, padding=3, fmt='%1.3f')
        ax_2.bar_label(rects_1, padding=3,fmt='%1.1f')
        ax_2.bar_label(rects_2, padding=3,fmt='%1.1f')
        ax_2.set_yticks([20,40,60,80,100])
        ax_2.set
    else:
        rects_1 = ax.bar(x - 0.5*width, own_auc, width, label = 'Own Auc', color = 'black')
        rects_2 = ax.bar(x + 0.5*width, MVTechAD_auc, width, label = 'MVTechAD Auc', color = 'grey')
        ax.set_ylabel('Auccarcy')
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.legend()
        ax.bar_label(rects_1, padding=3,fmt='%1.1f')
        ax.bar_label(rects_2, padding=3,fmt='%1.1f')
        ax.set_yticks([20,40,60,80,100])
    
    fig.tight_layout()

    if save_fig:
        file_name = str(int(time.time())) + title.replace(' ', '_') + '_' + '.svg'
        if not os.path.exists(res_path):
            os.makedirs(res_path) 
        plt.savefig(os.path.join(res_path,'plots', file_name), bbox_inches = 'tight')
        
    if show:
        plt.show()
        
def extract_vals_for_plot(summary_df: pd.DataFrame):
    '''
    Takes pandas DataFrame and extracts values for plotting
    '''
    labels = np.array(summary_df.index, dtype=str)
    feature_extraction = summary_df.loc[:, 'feature_extraction'].values
    embedding = summary_df.loc[:, 'embedding_of_feature'].values
    search = summary_df.loc[:, 'calc_distances'].values
    calc_distances = summary_df.loc[:, 'calc_scores'].values
    own_auc = summary_df.loc[:, 'img_auc_own'].values*100
    MVTechAD_auc = summary_df.loc[:, 'img_auc_MVTechAD'].values*100
    storage = summary_df.loc[:, 'backbone_storage'].values
    coreset_size = get_coreset_size_length_inner_process(summary_df) 
    
    return labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, coreset_size

def remove_failed_run_dirs(failed_runs: np.ndarray):
    '''
    removes failed runs from run_dirs
    '''
    dir_path = os.path.dirname(os.path.abspath(__file__))
    for folder in failed_runs:
        path = os.path.join(dir_path,'results', folder)
        if os.path.isdir(path):
            shutil.rmtree(path)
    return None

def remove_all_empty_run_dirs():
    '''
    removes all empty run dirs
    '''
    counter = 0
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    for folder in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, folder, 'csv')):
            if len(os.listdir(os.path.join(dir_path, folder, 'csv'))) == 0:
                counter += 1 
                shutil.rmtree(os.path.join(dir_path, folder))
    print(f'Removed {counter} empty folders')
    return None

def remove_uncomplete_runs(main_dir = r'/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_PatchCore/'):
    '''
    checks if all csv files are complete and removes uncomplete runs
    '''
    counter = 0
    dir_path = os.path.join(main_dir, 'results')
    for folder in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, folder, 'csv')):
            if len(os.listdir(os.path.join(dir_path, folder, 'csv'))) == 0:
                counter += 1 
                shutil.rmtree(os.path.join(dir_path, folder))
            else:
                for file in os.listdir(os.path.join(dir_path, folder, 'csv')):
                    if file.startswith('summary'):
                        try:
                            summary_df = pd.read_csv(os.path.join(dir_path, folder, 'csv', file), index_col=0)
                        except:
                            counter += 1 
                            shutil.rmtree(os.path.join(dir_path, folder))
                            break
                        if summary_df.shape[1] != int(16):
                            counter += 1 
                            shutil.rmtree(os.path.join(dir_path, folder))
                            break
    print(f'Removed {counter} empty folders')
    return None

def get_coreset_size_length_inner_process(pd_summary):
    '''
    some string operations to get the number of features used (length), returns int
    '''
    result = []
    for k in range(pd_summary.shape[0]):
        try:
            b = pd_summary.loc[:,'coreset_size'].values[0]
            res = b[b.find(' ')+1:b.find(')')]
        except:
            res = 'NA'
    
    print(res)
    return res

def sort_by_attribute(attribute, labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage):
    '''
    returns a sorted dataframe by attribute.
    Give as attribute a copy of one of the other arguments.
    '''
    order = np.argsort(attribute)
    return labels[order], feature_extraction[order], embedding[order], search[order], calc_distances[order], own_auc[order], MVTechAD_auc[order], storage[order]

def filter_by_contain_in_label_str(labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage ,to_contain: list, to_delete: list):
    '''
    returns a list of labels that contain to_contain and do not contain to_delete
    '''
    for pattern in to_contain:
        mask_1 = [True if label.__contains__(pattern) else False for label in labels]
        labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage = labels[mask_1], feature_extraction[mask_1], embedding[mask_1], search[mask_1], calc_distances[mask_1], own_auc[mask_1], MVTechAD_auc[mask_1], storage[mask_1]

    for pattern in to_delete:
        mask_2 = [True if not label.__contains__(pattern) else False for label in labels]
        labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage = labels[mask_2], feature_extraction[mask_2], embedding[mask_2], search[mask_2], calc_distances[mask_2], own_auc[mask_2], MVTechAD_auc[mask_2], storage[mask_2]
    return labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage

def shorten_labels(labels, to_delete: list):
    '''
    returns a list of labels with shortened names
    '''
    for k in range(len(to_delete)):
        labels = [label.replace(to_delete[k],'') for label in labels]
    return labels

def get_plot_ready_data(this_run_id, res_path, to_contain, to_delete, take_n_best = None): 
    '''
    returns data, that is ready to be plotted. Specify filters by the to_contain and to_delete lists. optional.
    '''
    summary_pd = get_summary_df(this_run_id, res_path)
    labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, coreset_size = extract_vals_for_plot(summary_pd)
    for k in range(len(coreset_size)):
        labels[k] = labels[k] + '\n(' + str(coreset_size[k]) + ')'
    print('Raw: #', len(labels))
    # if attribute_to_sort_by is not None:
    labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage = sort_by_attribute(MVTechAD_auc, labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage)
    if take_n_best is not None:
        labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage = labels[-take_n_best:], feature_extraction[-take_n_best:], embedding[-take_n_best:], search[-take_n_best:], calc_distances[-take_n_best:], own_auc[-take_n_best:], MVTechAD_auc[-take_n_best:], storage[-take_n_best:]
    
    labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage = filter_by_contain_in_label_str(labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage, to_contain=to_contain, to_delete=to_delete)
    print('Filtered: #',len(labels))

    labels = shorten_labels(labels, to_delete=to_contain) # and mention in title of plot instead in order to keep somehow short labels
    for k in range(len(labels)):
        labels[k] = labels[k].replace('-','\n')
    return labels, feature_extraction, embedding, search, calc_distances, own_auc, MVTechAD_auc, storage

def remove_test_dir():
    '''
    removes test dir if it is bigger than 5GB
    '''
    if get_dir_size(os.path.join(os.getcwd(), 'test'))/(1024*1024*1024) > 5:
        print('delete')
        # os.remove(os.path.join(os.getcwd(), 'test', 'test.txt'))
        try:
            shutil.rmtree(os.path.join(os.getcwd(), 'test'))
            print('deleted')
        except:
            print('could not delete')  
            
def get_dir_size(path='.'):
    '''
    returns size of directory in bytes
    '''
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return np.divide(e_x, np.sum(e_x)) 