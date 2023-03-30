import numpy as np
from skimage.measure import label
from seg_utils import get_mask, get_gt, create_pmask, filter_pmask, get_pseudo_ground_truths, f1_score


def get_f1_scores(datapath, sample_ids, methods, weights, agree_ratio, radius):
    #get probability masks
    pmasks = {}
    for sid in sample_ids:
        pmasks[sid] = {}
        for method in methods:
            method_mask = get_mask(datapath, sid, method)
            pmasks[sid][method] = create_pmask(method_mask, radius)
            
    #get pseudo-ground-truths
    pseudo_gts = get_pseudo_ground_truths(pmasks, sample_ids, methods, weights, agree_ratio)
    
    #filter probability masks
    f_pmasks = {}
    for sid in sample_ids:
        f_pmasks[sid] = {}
        for method in methods:
            f_pmasks[sid][method] = filter_pmask(pmasks[sid][method], label(pseudo_gts[sid]))
    
    #re-generate pseudo-ground-truths
    f_pseudo_gts = get_pseudo_ground_truths(f_pmasks, sample_ids, methods, weights, agree_ratio)
    
    #calculate average F1-score per method across samples
    f1_per_method = []
    for method in methods:
        mean_method_f1s = np.mean([f1_score(f_pseudo_gts[sid], label(f_pmasks[sid][method])) for sid in sample_ids])
        f1_per_method.append(mean_method_f1s)
        
    return f1_per_method