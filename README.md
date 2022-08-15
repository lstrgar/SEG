# seg-eval
The `eval.py` script does all the work here. This depends on a config yaml file in this case just called `config.yaml`. The data inputs are different segmentation masks for the same samples coming from different segmentation methods. The idea is to develop an ensemble based approach to approximate a "ground truth" label for the segmentation against which each individual mask can be evaluated. 

If you open the config file you'll see a list of methods and a specified data directory. The masks for each method are assumed to live under `${data_dir}/${method}` and across methods each mask should share a commond name reflective of the underlying data sample it can be associated with. For an example see this directory on exacloud: ``/home/groups/ChangLab/dataset/HMS-TMA-TNP/DATA-03292022`. This is an example of actual inputs / outputs from running this script. 

Two other important parameters in the config file are `num_agree` and `radii`. The former corresponds to the number of methods that must agree on a pixel (cell or not) to be called a true positive. The later is a hyper-parameter that indicates the amount of dilation about the centroid used to construct the probability segmentation masks. 

In the `eval.py` script you'll see three command line flags that can be passed. To run the entire script you should pass all three. If you run them separately they must be computed in serial. 

For more information you may refer to the following poster -- 


[HTAN_Abstract_TNP-TMA_analysis_final.pdf](https://github.com/lstrgar/seg-eval/files/9339751/HTAN_Abstract_TNP-TMA_analysis_final.pdf)
