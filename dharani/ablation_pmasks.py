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

files = sorted(os.listdir("/home/groups/ChangLab/dharani/OHSU-TMA/adjusted_ratios/results_mscx_overall/pmasks"))
path = "/home/groups/ChangLab/dharani/OHSU-TMA/adjusted_ratios/results_mscx_overall/pmasks/"
output = "/home/groups/ChangLab/dharani/OHSU-TMA/adjusted_ratios/results_msc_overall/pmasks"

if not os.path.exists(output):
    os.makedirs(output)

# m - mesmer, s - stardist, r - maskrcnn, c - cellpose, u - unet, x - unmicst

for file in tqdm(files):
    #print(file)
    with open(os.path.join(path, file), "rb") as f:
        pmasks_a10 = pickle.load(f)
    
    masks = {}
    methods = ["Mesmer", "Stardist", "Cellpose"]
    #rmasks = {}

    for keys, values in pmasks_a10.items():
        rmasks = {}
        for method, array in values.items():
            if method in methods:
                rmasks[method] = array
    
        stack = np.stack(list(rmasks.values()))
        avg_mask = stack.sum(0)
        rmasks["sum"] = avg_mask
        masks[keys] = rmasks
    
    with open(os.path.join(output, os.path.basename(file)), "wb") as handle:
        pickle.dump(masks, handle)