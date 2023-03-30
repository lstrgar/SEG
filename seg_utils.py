import os
import numpy as np
from skimage.io import imread
from skimage.morphology import binary_dilation, disk
from skimage.measure import regionprops, label
from skimage.segmentation import expand_labels


def get_mask(datapath, sid, method):
    datapath = os.path.join(datapath,method)
    if method == 'Mesmer': datapath = os.path.join(datapath, 'mesmer_nuc')
    if method == 'Stardist': datapath = os.path.join(datapath, 'vf')
    files = os.listdir(datapath)
    sid_file = [f for f in files if sid in f][0]
    mask_path = os.path.join(datapath, sid_file)
    return imread(mask_path)


def get_gt(sid):
    label_dir = '/home/groups/ChangLab/dataset/HMS-TMA-TNP/OHSU-TMA/Labels'
    f = [f for f in os.listdir(label_dir) if sid in f][0]
    return imread(os.path.join(label_dir, f))


def create_pmask(seg_mask, r):
    """converts a standard segmentation mask into one with a dilated area around 
    the center of the original mask"""
    mask = np.zeros_like(seg_mask)
    rps = regionprops(seg_mask)
    
    #get center of each cell and set the value to one
    centroids = np.array(
        list(map(lambda x : np.array(x.centroid).astype(int), rps))
    )
    mask[centroids[:,0], centroids[:,1]] = 1
    
    #dilate around the center of the cell with the radius size and binarize the mask
    #mask = binary_dilation(mask, disk(r))
    mask = expand_labels(label(mask), distance=r)
    mask = (mask > 0).astype('int')
    
    return mask


def filter_pmask(mask, avg_labs):
    """filter probability mask for a single method (mask) using the thresholded, averaged, proability mask
    such that the new probability mask contains cell regions that only overlap with one cell region in the
    pseudo-ground-truth"""
    #copy 
    filtered = mask.copy()
    mask_lab = label(mask)
    rps = regionprops(mask_lab)
    
    for rp in rps: #for each cell region
        #get cell center coordinates from single-method mask
        coords = rp.coords
        #use those coords to get values of averaged mask 
        vals = avg_labs[coords[:,0], coords[:,1]]
        uniq, counts = np.unique(vals, return_counts=True)
        
        #ignore background
        if uniq[0] == 0:
            uniq = uniq[1:]
            counts = counts[1:]
            
        n_unique = len(uniq)
        
        #if more than 1 value, zero out pixels in mask that do not equal the most common value
        if n_unique > 1:
            amax = np.argmax(counts)
            top_val = uniq[amax]
            idxs = np.where(vals != top_val)
            to_zero = coords[idxs,:][0]
            filtered[to_zero[:,0], to_zero[:,1]] = False
            
    return filtered


def f1_score(gt, m):
    """returns precision and recall for a pair of masks"""
    rps = regionprops(m)
    coords = list(map(lambda x : x.coords, rps))
    correct = 0
    
    for c in coords:
        correct += (gt[c[:,0], c[:,1]]).max()
    precision = correct / len(rps)
    
    gt_labs = label(gt)
    gt_rps = regionprops(gt_labs)
    coords = list(map(lambda x : x.coords, gt_rps))
    correct = 0
    
    for c in coords:
        correct += (m[c[:,0], c[:,1]]).max() > 0
    recall = correct / len(gt_rps)
    
    assert precision <= 1
    assert precision >= 0
    assert recall <= 1
    assert recall >= 0
    
    f1 = lambda p,r: 2 * (p * r) / (p + r)

    return f1(precision, recall)



def get_pseudo_ground_truths(pmasks, sample_ids, methods, weights, agree_ratio):
    pseudo_gts = {}

    for s,sid in enumerate(sample_ids):
        gt = sum([pmasks[sid][method] * weights[s][m] for m,method in enumerate(methods)])
        pseudo_gts[sid] = (gt >= agree_ratio).astype('int')
    
    return pseudo_gts



