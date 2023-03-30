import os
import numpy as np
from scipy.special import softmax
from skimage.measure import label
from seg_utils import get_mask, get_gt, create_pmask, filter_pmask, f1_score

def get_pseudo_ground_truths(pmasks, sample_ids, methods, weights, agree_ratio, filtered=False):
    pseudo_gts = {}
    for s,sid in enumerate(sample_ids):
        pseudo_gts[sid] = {}
        for method_out in methods:
            if filtered:
                pmasks_ = pmasks[method_out][sid] #filtered with resepect to method-left-out
            else:
                pmasks_ = pmasks[sid]
            methods_in = [m for m in methods if m != method_out]
            gt = sum([pmasks_[method]*weights[s][m] for m,method in enumerate(methods_in)])
            pseudo_gts[sid][method_out] = (gt >= agree_ratio).astype('int')   
    return pseudo_gts

def get_weights(datapath, sample_ids, methods, radius, agree_ratio): 
    # get probability masks
    pmasks = {}
    for sid in sample_ids:
        pmasks[sid] = {}
        for method in methods:
            method_mask = get_mask(datapath, sid, method)
            pmasks[sid][method] = create_pmask(method_mask, radius)
               
    #get pseudo-ground-truths for each method-left-out
    weights = np.ones((len(sample_ids), len(methods) - 1)) * (1/ (len(methods) - 1))
    pseudo_gts = get_pseudo_ground_truths(pmasks, sample_ids, methods, weights, agree_ratio)
    
    #filter probability masks
    f_pmasks = {}
    for method_out in methods:
        f_pmasks[method_out] = {}
        for sid in sample_ids:
            f_pmasks_ = {}
            for method_in in methods:
                if method_in == method_out: continue 
                f_pmasks_[method_in] = filter_pmask(pmasks[sid][method_in], label(pseudo_gts[sid][method_out]))
            f_pmasks[method_out][sid] = f_pmasks_
            
    #regenerate pseudo-ground-truths
    f_pseudo_gts =  get_pseudo_ground_truths(f_pmasks, sample_ids, methods, weights, agree_ratio, filtered=True)
    
    #generate F1 table (N X N), N=# of methods
    f1_table = np.zeros((len(sample_ids), len(methods), len(methods)))
    for s,sid in enumerate(sample_ids):
        for i,method_in in enumerate(methods):
            for j,method_out in enumerate(methods):
                if method_out == method_in: 
                    f1_table[s][i][j] = -999
                    continue
                method_mask = label(f_pmasks[method_out][sid][method_in])
                f1_table[s][i][j] = f1_score(f_pseudo_gts[sid][method_out], method_mask)  
                
    """ 
    #convert f1 table to delta table
    f1_delta_table = np.zeros((len(sample_ids), len(methods), len(methods)))
    for s,_ in enumerate(f1_table):
        for i,f1_i in enumerate(f1_table[s]):
            mean_f1 = np.mean([f1 for f1 in f1_i if f1 != -999])
            for j,f1 in enumerate(f1_i):
                if i == j: 
                    f1_delta_table[s][i][j] = -999
                    continue
                f1_delta_table[s][i][j] = 100 * (f1 - mean_f1) / mean_f1
    
    #calculate weights
    weights  = np.zeros((len(sample_ids),len(methods)))
    avg_deltas  = np.zeros((len(sample_ids),len(methods)))
    for s,sid in enumerate(sample_ids):
        delta_table = f1_delta_table[s]
        for m, method in enumerate(methods):
            avg_deltas[s][m] = -np.mean([f1_d for f1_d in delta_table[:,m] if f1_d != -999])
        weights[s] = softmax(avg_deltas[s])
    """
    weights  = np.zeros((len(sample_ids),len(methods))) 
    for s,sid in enumerate(sample_ids):
        f1_t = f1_table[s]
        f1_t[f1_t == -999] = np.nan
        mF1 = np.nanmean(f1_t, axis=1)
        mean_mF1 = np.mean(mF1)
        z = 100 * (mF1 - mean_mF1) / mean_mF1
        weights[s] = softmax(z)
       
        
    return weights









    
