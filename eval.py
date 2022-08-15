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
import time

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

def compute_masks(sid, radii, methods, datapath):
    masks = {}
    for r in radii:
        rmasks = {}
        t0 = time.time()
        for m in methods:
            segpath = glob(os.path.join(datapath, m, sid + "*"))[0]
            pmask = create_pmask(segpath, r)
            rmasks[m] = pmask
        t1 = time.time()
        print(f"SEPARATE MASKS -- {sid} // {r} // {np.round(t1 - t0, 2)} sec")
        t0 = time.time()
        stack = np.stack(list(rmasks.values()))
        avg_mask = stack.mean(0)
        t1 = time.time()
        print(f"COMPUTER MEAN -- {sid} // {r} // {np.round(t1 - t0, 2)} sec")
        rmasks["mean"] = avg_mask
        masks[r] = rmasks
    return masks

def write_pmasks(sample_ids, radii, methods, datapath, save_dir):
    print(f"Writing probability masks for {len(sample_ids)} samples")
    for sid in tqdm(sample_ids):
        save_path = os.path.join(save_dir, f"{sid}.pkl")
        if os.path.exists(save_path):
            continue

        sid_masks = compute_masks(sid, radii, methods, datapath)
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
            avg = masks["mean"]
            avg_threshd = (avg >= (min_num_agree / len(methods)))
            avg_labs = label(avg_threshd)

            r_filtered_masks = {}
            for m in methods:
                r_filtered_masks[m] = filter_mask(masks[m], avg_labs)

            new_stack = np.stack(list(r_filtered_masks.values()))        
            new_avg = new_stack.mean(0)
            r_filtered_masks["mean"] = new_avg
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


def evaluate_masks(sample_ids, filtered_pmask_save_dir, radii, min_num_agree, num_methods):
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

            avg = masks["mean"]
            avg_thresh = (avg >= (min_num_agree / num_methods))
            
            for name, mask in masks.items():
                if name == "mean":
                    continue
                labd_mask = label(mask)
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

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    pmask_save_dir = os.path.join(results_dir, "pmasks")
    if not os.path.exists(pmask_save_dir):
        os.makedirs(pmask_save_dir)

    sample_ids = os.listdir(os.path.join(datapath, methods[0]))
    sample_ids = [s.split(".")[0] for s in sample_ids]

    if args.compute_pmasks:
        write_pmasks(sample_ids, radii, methods, datapath, pmask_save_dir)

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