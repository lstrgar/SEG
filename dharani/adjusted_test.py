import numpy as np
from skimage.io import imread
import os
import copy
import pickle
from skimage.measure import regionprops, label

with open("/home/groups/ChangLab/dharani/OHSU-TMA/adjusted_ratios/results_mscx/pmasks/BR1506-A015 - Scene 002.pkl", "rb") as handle:
    data = pickle.load(handle)

#print (data)

for r, masks in data.items():
    avg = masks["sum"]
    print("r", r)
    for name, mask in masks.items():
        if name == "sum":
            print("Sum", np.unique(mask))
        #continue
        #print(f"name {name}")
        #print(np.unique(mask))
        #print("mask", mask)
        #labd_mask = label(mask)
        #print(f"Label Mask {labd_mask}")
        #print("values label", np.unique(labd_mask))
        #rps = regionprops(labd_mask)
        #print(f"rps {rps}")