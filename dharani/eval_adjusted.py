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
    print("Working on creating pmasks")
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
            print("Max pmask", np.max(pmask))
            for key, value in adjusted_ratios.items():
                for radius, ratio in value.items():
                    if key == m and radius == r:
                        print(f"Methods {key}, radius {radius}, ratio {ratio}")
                        temp_ratio = ratio
            print("Multiply by", temp_ratio)
            print("Max pmask adjust", np.max(pmask * temp_ratio))
            rmasks[m] = pmask * temp_ratio
        #stack = np.stack(list(rmasks.values()))
        #avg_mask = stack.mean(0)
        #rmasks["mean"] = avg_mask
        stack = np.stack(list(rmasks.values()))
        sum = stack.sum(0)
        rmasks["sum"] = sum
        print(f"Sum unique values - {np.unique(sum)}")
        masks[r] = rmasks
        print(masks)
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
    mask_lab = label(mask)
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
            filtered[to_zero[:,0], to_zero[:,1]] = False
    return filtered


def filter_pmasks(sample_ids, pmask_save_dir, filtered_save_dir, min_num_agree, methods):
    print(f"Filtering probability masks for {len(sample_ids)} samples")
    for sid in tqdm(sample_ids):
        with open(os.path.join(pmask_save_dir, f"{sid}.pkl"), "rb") as handle:
            data = pickle.load(handle)

        filtered_masks = {}
        for r, masks in data.items():
            avg = masks["sum"]
            avg_threshd = (avg >= (min_num_agree / len(methods)))
            avg_labs = label(avg_threshd)

            r_filtered_masks = {}
            for m in methods:
                r_filtered_masks[m] = filter_mask(masks[m], avg_labs)

            new_stack = np.stack(list(r_filtered_masks.values()))        
            new_avg = new_stack.sum(0)
            r_filtered_masks["sum"] = new_avg
            filtered_masks[r] = r_filtered_masks
            
        with open(os.path.join(filtered_save_dir, f"{sid}.pkl"), "wb") as handle:
            pickle.dump(filtered_masks, handle)
            
    print(f"Filtered probability masks saved to {filtered_save_dir}")


def eval_mask(gt, m):
    rps = regionprops(m)
    print(f"rps {len(rps)}")
    coords = list(map(lambda x : x.coords, rps))
    correct = 0
    
    for c in coords:
        correct += (gt[c[:,0], c[:,1]]).max()
    precision = correct / len(rps)
    
    gt_labs = label(gt)
    print(f"Label Mask recall {np.unique(gt_labs)}")
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


def evaluate_masks(sample_ids, filtered_pmask_save_dir, radii, min_num_agree, num_methods):
    precision = {}
    recall = {}
    print(f"Computing precision and recall for {len(sample_ids)} samples")
    for sid in tqdm(sample_ids):
        data_load_path = os.path.join(filtered_pmask_save_dir, f"{sid}.pkl")
        with open(data_load_path, "rb") as handle:
            data = pickle.load(handle)
        
        #print(data)

        sid_precisions = dict((r, {}) for r in radii)
        sid_recalls = dict((r, {}) for r in radii)

        for r, masks in data.items():

            avg = masks["sum"]
            print(f"sum {np.unique(avg)}")
            avg_thresh = (avg >= (min_num_agree / num_methods))
            print(f"Label Mask {np.unique(avg_thresh)}")
            for name, mask in masks.items():
                if name == "sum":
                    continue
                print("values mask", np.unique(mask))
                mask_thresh = (mask > 0) # = (min_num_agree / num_methods))
                labd_mask = label(mask_thresh)
                print("values label", np.unique(labd_mask))
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
    num_agree = config["num_agree"]
    datapath = config["datapath"]
    results_dir = config["resultsdir"]
    #adjusted_ratios = config["adjusted_ratios"]
    
    #adjusted_ratios = {"Mesmer": {4:0.04, 8:0.28, 12:0.76, 16:0.88, 20:0.85, 24:0.49, 28:0.09, 32:0.11}, "Stardist": {4:0.59, 8:0.53, #12:0.11, 16:0.01, 20:0.01, 24:0.01, 28:0.01, 32:0.15}, "Cellpose":{4:0.06, 8:0.12, 12:0.10, 16:0.10, 20:0.12, 24:0.28, 28:0.27, 32:0.20}, #"UnMicst": {4:0.31, 8:0.05, 12:0.01, 16:0.01, 20:0.01, 24:0.20, 28:0.62, 32:0.53}} # recall
    
    #adjusted_ratios = {"Mesmer": {4:0.21, 8:0.24, 12:0.38, 16:0.46, 20:0.40, 24:0.27, 28:0.18, 32:0.15}, "Stardist": {4:0.26, 8:0.28, #12:0.20, 16:0.11, 20:0.07, 24:0.06, 28:0.06, 32:0.07}, "Cellpose":{4:0.23, 8:0.24, 12:0.27, 16:0.26, 20:0.34, 24:0.37, 28:0.43, 32:0.58}, #"UnMicst": {4:0.28, 8:0.23, 12:0.13, 16:0.15, 20:0.16, 24:0.28, 28:0.31, 32:0.18}} # overall
 
    adjusted_ratios = {"Mesmer": {4:0.25, 8:0.25, 12:0.25, 16:0.25, 20:0.25, 24:0.25, 28:0.25, 32:0.25}, "Stardist":{4:0.25, 8:0.25, 12:0.25, 16:0.25, 20:0.25, 24:0.25, 28:0.25, 32:0.25}, "Cellpose":{4:0.25, 8:0.25, 12:0.25, 16:0.25, 20:0.25, 24:0.25, 28:0.25, 32:0.25}, "UnMicst": {4:0.25, 8:0.25, 12:0.25, 16:0.25, 20:0.25, 24:0.25, 28:0.25, 32:0.25}} # equal weights

    #adjusted_ratios = {"Mesmer": {4:0.45, 8:0.12, 12:0.07, 16:0.04, 20:0.06, 24:0.12, 28:0.21, 32:0.10}, "Stardist": {4:0.05, 8:0.09, #12:0.12, 16:0.17, 20:0.25, 24:0.21, 28:0.19, 32:0.01}, "Cellpose":{4:0.40, 8:0.27, 12:0.24, 16:0.13, 20:0.23, 24:0.36, 28:0.44, 32:0.85}, #"UnMicst": {4:0.10, 8:0.50, 12:0.55, 16:0.64, 20:0.45, 24:0.29, 28:0.14, 32:0.03}} #precision
    
    print(f"Ratios {adjusted_ratios}")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    pmask_save_dir = os.path.join(results_dir, "pmasks")
    if not os.path.exists(pmask_save_dir):
        os.makedirs(pmask_save_dir)

    sample_ids = sorted(os.listdir(os.path.join(datapath, methods[0])))
    sample_ids = [s.split(".")[0] for s in sample_ids]
    #sample_ids = [sample_ids[0]]
    print(f"File used is - {sample_ids}")
    
    if args.compute_pmasks:
        write_pmasks(sample_ids, radii, methods, datapath, pmask_save_dir, adjusted_ratios)

    filtered_pmask_save_dir = os.path.join(results_dir, "filtered_pmasks")
    if not os.path.exists(filtered_pmask_save_dir):
        os.makedirs(filtered_pmask_save_dir)

    if args.filter_pmasks:
        filter_pmasks(sample_ids, pmask_save_dir, filtered_pmask_save_dir, num_agree, methods)

    precision_scores_path = os.path.join(results_dir, "precision_scores.pkl")
    recall_scores_path = os.path.join(results_dir, "recall_scores.pkl")

    if args.compute_scores:
        precision, recall = evaluate_masks(sample_ids, filtered_pmask_save_dir, radii, num_agree, len(methods))
        with open(precision_scores_path, "wb") as handle:
            pickle.dump(precision, handle)
            print(f"Saved precision scores to {precision_scores_path}")
        with open(recall_scores_path, "wb") as handle:
            pickle.dump(recall, handle)
            print(f"Saved recall scores to {recall_scores_path}")


if __name__ == '__main__':
    main()