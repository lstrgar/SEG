import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from skimage.io import imread, imsave
import cv2
import pickle
import tqdm

path = "/home/groups/ChangLab/dharani/OHSU-TMA/"

radii = [4, 8, 12, 16, 20, 24, 28, 32]
methods = ["Mesmer", "Stardist", "Cellpose", "UnMicst", "mean"]
           

# plot1 
#img1 = imread(os.path.join(path, "Contours/BR1506-A015 - Scene 002.png"))
#plt.imshow(img1[1000:1500, 1000:1500])
#plt.title("Contours")
#plt.show()

# plot 2
#img2 = imread(os.path.join(path, "Foreground/BR1506-A015 - Scene 002.png"))
#plt.imshow(img2[1000:1500, 1000:1500])
#plt.title("Foreground")
#plt.show()

# plot 3
#img3 = imread(os.path.join(path, "Labels/BR1506-A015 - Scene 002.tif"))
#plt.imshow(img3[1000:1500, 1000:1500])
#plt.title("Labels")
#plt.show()

# plot 4
#img4 = imread(os.path.join(path, "Originals/BR1506-A015 - Scene 002.png"))
#plt.imshow(img4[1000:1500, 1000:1500])
#plt.title("Originals")
#plt.show()

def to_binary(image):
    gd_img = image
    #gd_img_uint8 = np.uint8(gd_img) 
    gd_img_binary = (gd_img > 0).astype(int)
    print("Max value : ", np.max(gd_img_binary))
    return gd_img_binary

def dice_coeff(true, pred):
    true_f = true.flatten()
    pred_f = pred.flatten()
    union = np.sum(true_f) + np.sum(pred_f)
    if union==0:
        return 1
    intersection = np.sum(true_f * pred_f)
    return (2.* intersection/union)
    
def jaccard_idx(true, pred):
    intersection = (true * pred).sum()
    union = true.sum() + pred.sum() #- intersection
    return intersection / union
    
path_pmasks = os.path.join(path, "results_mscx/pmasks")
files = sorted(os.listdir(path_pmasks))

for file in files:
    with open(os.path.join(path_pmasks, file), 'rb') as f:
        values = pickle.load(f)
    
    print(f"Pickle file name - {file}")
    
    base_name = file.split('.')[0]
    groundtruth = str(base_name) + ".tif"
    print("Label file {}".format(os.path.join(path, "Labels", groundtruth)))
    
    img3 = imread(os.path.join(path, "Labels", groundtruth))
    gd_binary = to_binary(img3)
    
    for keys, value in values.items():
        if keys in radii:
            for methods, array in value.items():
                if methods == 'Mesmer':
                    print(np.unique(array))
                
                    dice_score = dice_coeff(gd_binary, array.astype(int))
                    print("Dice score of radii {} is {}".format(keys, dice_score))
                
                    iou_score = jaccard_idx(gd_binary, array.astype(int))
                    print("IOU score of radii {} is {}".format(keys, iou_score))