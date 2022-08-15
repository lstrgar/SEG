from argparse import ArgumentParser
import yaml
import numpy as np
from skimage.io import imread
from skimage.morphology import binary_dilation, disk
from skimage.measure import regionprops, label
import os
import copy
from glob import glob
import pickle
from tqdm import tqdm

# make sure to uncomment the sample ids 

def getid(samp_path):
    return samp_path.split("/")[-1].split(".")[0]

def create_pmask(segpath, r):
    seg = imread(segpath)
    mask = np.zeros_like(seg)
    rps = regionprops(seg)
    centroids = np.array(
        list(map(lambda x : np.array(x.centroid).astype(int), rps))
    )
    mask[centroids[:,0], centroids[:,1]] = 1
    mask = binary_dilation(mask, disk(r))
    return mask 

def compute_masks(sid, radii, methods, datapath, adjusted_ratios):
    masks = {}
    for r in radii:
        rmasks = {}
        for m in methods:
            temp_ratio = 0.0
            segpath = glob(os.path.join(datapath, m, sid + "*"))[0]
            pmask = create_pmask(segpath, r)
            pmask = pmask.astype(int)
            temp_ratio = adjusted_ratios[m][r]
            rmasks[m] = pmask * temp_ratio
        stack = np.stack(list(rmasks.values()))
        avg = stack.sum(0)
        rmasks["avg"] = avg
        masks[r] = rmasks
    return masks

def write_pmasks(sample_ids, radii, methods, datapath, save_dir, adjusted_ratios):
    print(f"Writing probability masks for {len(sample_ids)} samples")
    for sid in tqdm(sample_ids):
        save_path = os.path.join(save_dir, f"{sid}.pkl")
        if os.path.exists(save_path):
            continue

        sid_masks = compute_masks(sid, radii, methods, datapath, adjusted_ratios)
        with open(save_path, "wb") as handle:
            pickle.dump(sid_masks, handle)
    print(f"Proability masks saved to {save_dir}")


def filter_mask(mask, avg_labs):
    filtered = copy.deepcopy(mask)
    mask_lab = label(mask > 0)
    rps = regionprops(mask_lab)
    for rp in rps:
        coords = rp.coords
        vals = avg_labs[coords[:,0], coords[:,1]]
        uniq, counts = np.unique(vals, return_counts=True)
        if uniq[0] == 0:
            uniq = uniq[1:]
            counts = counts[1:]
        n_unique = len(uniq)
        if n_unique > 1:
            amax = np.argmax(counts)
            top_val = uniq[amax]
            idxs = np.where(vals != top_val)
            to_zero = coords[idxs,:][0]
            filtered[to_zero[:,0], to_zero[:,1]] = 0
    return filtered


def filter_pmasks(sample_ids, pmask_save_dir, filtered_save_dir, thresh, methods, op):
    print(f"Filtering probability masks for {len(sample_ids)} samples")
    for sid in tqdm(sample_ids):
        with open(os.path.join(pmask_save_dir, f"{sid}.pkl"), "rb") as handle:
            data = pickle.load(handle)

        filtered_masks = {}
        for r, masks in data.items():
            avg = masks["avg"]
            avg_threshd = eval("avg" + op + "thresh")
            avg_labs = label(avg_threshd)

            r_filtered_masks = {}
            for m in methods:
                r_filtered_masks[m] = filter_mask(masks[m], avg_labs)

            new_stack = np.stack(list(r_filtered_masks.values()))    
            new_avg = new_stack.sum(0)
            r_filtered_masks["avg"] = new_avg
            filtered_masks[r] = r_filtered_masks
            
        with open(os.path.join(filtered_save_dir, f"{sid}.pkl"), "wb") as handle:
            pickle.dump(filtered_masks, handle)
            
    print(f"Filtered probability masks saved to {filtered_save_dir}")


def eval_mask(gt, m):
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
    
    return precision, recall


def evaluate_masks(sample_ids, filtered_pmask_save_dir, radii, thresh, op):
    precision = {}
    recall = {}
    print(f"Computing precision and recall for {len(sample_ids)} samples")
    for sid in tqdm(sample_ids):
        data_load_path = os.path.join(filtered_pmask_save_dir, f"{sid}.pkl")
        with open(data_load_path, "rb") as handle:
            data = pickle.load(handle)

        sid_precisions = dict((r, {}) for r in radii)
        sid_recalls = dict((r, {}) for r in radii)

        for r, masks in data.items():

            avg = masks["avg"]
            avg_thresh = eval("avg" + op + "thresh")
            for name, mask in masks.items():
                if name == "avg":
                    continue
                mask_thresh = (mask > 0)
                labd_mask = label(mask_thresh)
                prec, rec = eval_mask(avg_thresh, labd_mask)
                sid_precisions[r][name] = prec
                sid_recalls[r][name] = rec

        precision[sid] = sid_precisions
        recall[sid] = sid_recalls

    return precision, recall


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yml", help="Path to config file")
    parser.add_argument("--compute-pmasks", action="store_true", help="Compute probability masks")
    parser.add_argument("--filter-pmasks", action="store_true", help="Filter probability masks")
    parser.add_argument("--compute-scores", action="store_true", help="Compute scores")
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    methods = config["methods"]
    radii = config["radii"]
    thresh = config["thresh"]
    op = config["operator"]
    datapath = config["datapath"]
    results_dir = config["resultsdir"]
    
    #adjusted_ratios = {
    #    "Mesmer": {4:0.21, 8:0.24, 12:0.38, 16:0.46, 20:0.40, 24:0.27, 28:0.18, 32:0.15}, 
    #    "Stardist": {4:0.26, 8:0.28, 12:0.20, 16:0.11, 20:0.07, 24:0.06, 28:0.06, 32:0.07}, 
    #    "Cellpose":{4:0.23, 8:0.24, 12:0.27, 16:0.26, 20:0.34, 24:0.37, 28:0.43, 32:0.58}, 
    #    "UnMicst": {4:0.28, 8:0.23, 12:0.13, 16:0.15, 20:0.16, 24:0.28, 28:0.31, 32:0.18}
    #} # 5TMA
    
    #adjusted_ratios = {
    #    "Mesmer": {4:0.25, 8:0.25, 12:0.25, 16:0.25, 20:0.25, 24:0.25, 28:0.25, 32:0.25}, 
    #    "Stardist": {4:0.25, 8:0.25, 12:0.25, 16:0.25, 20:0.25, 24:0.25, 28:0.25, 32:0.25}, 
    #    "Cellpose":{4:0.25, 8:0.25, 12:0.25, 16:0.25, 20:0.25, 24:0.25, 28:0.25, 32:0.25}, 
    #    "UnMicst": {4:0.25, 8:0.25, 12:0.25, 16:0.25, 20:0.25, 24:0.25, 28:0.25, 32:0.25}
    #} # equal weights
    
    adjusted_ratios = {
         'unmicst': {2:0.15708386, 4:0.15110248, 6:0.08883816, 8:0.09821838, 10:0.13989789, 12:0.1258211, 14:0.12531543, 16:0.12113992},
         'unet':    {2:0.16780117, 4:0.15107227, 6:0.09650707, 8:0.08273092, 10:0.11099807, 12:0.12931626, 14:0.14245611, 16:0.15310841},
         'cellpose': {2:0.15878956, 4:0.14010026, 6:0.15456524, 8:0.13536751, 10:0.16678621, 12:0.16634452, 14:0.17150869, 16:0.16150663},
         'maskrcnn': {2:0.16786831, 4:0.18481555, 6:0.12052007, 8:0.20905416, 10:0.18547365, 12:0.18505647, 14:0.16851655, 16:0.17548371},
         'stardist': {2:0.17760816, 4:0.21399621, 6:0.29032945, 8:0.20934704, 10:0.14482307, 12:0.14180616, 14:0.1434568, 16:0.17070658},
         'mesmer': {2:0.17084894, 4:0.15891323, 6:0.24924, 8:0.26528199, 10:0.2520211, 12:0.25165549, 14:0.24874642, 16:0.21805476}}#88 TMA
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    pmask_save_dir = os.path.join(results_dir, "pmasks")
    if not os.path.exists(pmask_save_dir):
        os.makedirs(pmask_save_dir)

    sample_ids = sorted(os.listdir(os.path.join(datapath, methods[0])))
    sample_ids = [s.split(".")[0] for s in sample_ids]
    #sample_ids = sample_ids[:1]
    print(sample_ids)
    
    if args.compute_pmasks:
        write_pmasks(sample_ids, radii, methods, datapath, pmask_save_dir, adjusted_ratios)

    filtered_pmask_save_dir = os.path.join(results_dir, "filtered_pmasks")
    if not os.path.exists(filtered_pmask_save_dir):
        os.makedirs(filtered_pmask_save_dir)

    if args.filter_pmasks:
        filter_pmasks(sample_ids, pmask_save_dir, filtered_pmask_save_dir, thresh, methods, op)

    if args.compute_scores:
        precision, recall = evaluate_masks(sample_ids, filtered_pmask_save_dir, radii, thresh, op)
        precision_scores_path = os.path.join(results_dir, "precision_scores.pkl")
        recall_scores_path = os.path.join(results_dir, "recall_scores.pkl")
        with open(precision_scores_path, "wb") as handle:
            pickle.dump(precision, handle)
            print(f"Saved precision scores to {precision_scores_path}")
        with open(recall_scores_path, "wb") as handle:
            pickle.dump(recall, handle)
            print(f"Saved recall scores to {recall_scores_path}")


if __name__ == '__main__':
    main()
