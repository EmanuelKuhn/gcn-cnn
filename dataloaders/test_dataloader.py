#%%
%load_ext autoreload
%autoreload 2


#%%
%cd /home/emanuel/Documents/thesis/gcn-cnn/
#%%
!ls fp_data
#%%

import dataloader_triplet

from torchvision import transforms
from torch import nn

from argparse import Namespace

import pickle as pkl
# %%

opt = {
    "img_dir": None,
    "Channel25_img_dir": "fp_data/rplan12ChanImages",
    "batch_size": 8,
    "apn_dict_path": "fp_data/apn_dict_13K_pthres60.pkl",
    "use_box_feats": False,
    "hardmining": False,
    "use_25_images": True,
    "use_precomputed_25Chan_imgs": True,

}

opt = Namespace(**opt)
    
transform = nn.Sequential()

dataset = dataloader_triplet.RICO_TripletDataset(opt, transform)

# %%
batch = dataset.get_batch("train", 1)


# %%

import numpy as np

import torch

with np.load("fp_data/rplan12ChanImages/0.npz") as data:
    tensor = torch.tensor(data["arr_0"], dtype=torch.float32)

    # print(tensor.max())
# # %%

# def load_pickle(path):
#     with open(path, "rb") as f:
#         return pkl.load(f)

# fp_data = load_pickle("fp_data/FP_data.p")
# # %%
# def get_id_from_path(path):
#     return path.split("/")[-1].split(".")[0]

# splits = dict()

# for key in fp_data:
#     splits[key] = list(map(get_id_from_path, fp_data[key]))
# # %%
# splits.keys()

# def load_pickle(path):
#     with open(path, "rb") as f:
#         return pkl.load(f)

# fp_data = load_pickle("fp_data/FP_data.p")

# # %%
# fp_data
# # %%
# def get_id_from_path(path):
#     return path.split("/")[-1].split(".")[0]

# splits = dict()

# for key in fp_data:
#     splits[key] = list(map(get_id_from_path, fp_data[key]))
# # %%
# splits.keys()
# # %%
# all_ids = set(splits["train_fps"] + splits["val_fps"])
# # %%
# "0" in all_ids

# # %%

# "0" in splits["train_fps"]