from __future__ import print_function, division
#import ipdb
import os
import tensorflow as tf
from numpy.lib.function_base import gradient
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad
from torch.utils.data import DataLoader
from torchvision import transforms#, utils
from catalyst import utils
from sklearn.impute import SimpleImputer
from nilearn import image
import nilearn
from nilearn import plotting
import os
import nibabel as nib
import glob
from PIL import Image
from torch.utils.data.dataset import Dataset
import sklearn as skl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import KFold
from freesurfer_stats import CorticalParcellationStats
from catalyst import dl
from sklearn.metrics import mean_squared_error
from captum.attr import visualization as viz
from captum.attr import (
    Saliency, 
    IntegratedGradients,
    NoiseTunnel,
    LayerGradCam, 
    FeatureAblation, 
    LayerActivation, 
    LayerAttribution
)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

#creating dataset: set filepath to where data is
image_list = glob.glob(
    '/data/users2/vjoshi6/MRIDataConverted/128127_3T_T1w_MPR1.nii.gz')
stat_list_r = glob.glob(
    '/data/hcp-plis/hdd01/128127/T1w/128127/stats/rh.aparc.stats')
stat_list_l = glob.glob(
    '/data/hcp-plis/hdd01/128127/T1w/128127/stats/lh.aparc.stats')

#the code below joins the stat files and the images with brain region names into a dataframe
#getting both hemispheres and checking that they are matched up correctly
volume_list_r = []
volume_list_l = []
for i in range(len(stat_list_r)):
    #using volume path for right and left hemispheres to read in data for each hemisphere
    stats_r = CorticalParcellationStats.read(stat_list_r[i])
    stats_l = CorticalParcellationStats.read(stat_list_l[i])

    #getting numbers so I can properly label each gray matter measurement df
    path_r = stat_list_r[i].split('/')
    path_l = stat_list_l[i].split('/')
    subject_r = path_r[4]
    subject_l = path_l[4]


    #creating measurement dataframes and renaming them
    df_r = stats_r.structural_measurements[[
                       'gray_matter_volume_mm^3']]
    df_l = stats_l.structural_measurements[[
        'gray_matter_volume_mm^3']]

    df_r.rename(columns={'gray_matter_volume_mm^3': subject_r}, inplace=True)
    df_l.rename(columns={'gray_matter_volume_mm^3': subject_l}, inplace=True)

    keys_r_initial = np.array(stats_r.structural_measurements[['structure_name']])
    keys_l_initial = np.array(stats_l.structural_measurements[['structure_name']])

    keys_r = []
    keys_l = []
    for keys in keys_r_initial:
        keys_r.append(keys[0])
    
    for keys in keys_l_initial:
        keys_l.append(keys[0])

    df_r['name'] = keys_r
    df_l['name'] = keys_l

    df_r = df_r.set_index('name')
    df_l = df_l.set_index('name')

    volume_list_r.append(df_r)
    volume_list_l.append(df_l)

#this just joins the left and right hemisphere volumes
volume_list = []
for i in range(len(volume_list_l)):
    if (volume_list_l[i].columns == volume_list_r[i].columns):
        left_values = volume_list_l[i].iloc[:, 0]
        right_values = volume_list_r[i].iloc[:, 0]
        values = left_values.append(right_values)
        values = pd.DataFrame(values)
        volume_list.append(values)
    else:
        print('ERROR')
        print(volume_list_l[i])
        break


#this for loop just makes it so the images and the labels are alligned with
#the same indexes
volume_list_ordered = []
for i in range(len(image_list)):
    image1 = image_list[i].split('/')[5][0:6]
    for value in volume_list:
        if (value.columns == image1):
            #value = value.values
            #value = value.reshape(1,68)
            volume_list_ordered.append(value)


d = {'images': image_list, 'gray_matter': volume_list_ordered}
df = pd.DataFrame(d)
images_as_np = []


#gets images finalized and preprocessed
for i in range(len(image_list)):
    single_image_path = image_list[i]
    image_name = str(single_image_path)
    #open that image
    img = nib.load(image_name)
    #convert to numpy and preprocess to get into tensor
    img = img.get_fdata().astype('float32')
    img = (img - img.min()) / (img.max() - img.min())
    #labels = volume_list_ordered[i].iloc[0]
    #img, temp = nobrainer.volume.apply_random_transform_scalar_labels(img, labels)
    new_img = np.zeros(img.shape)
    new_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img
    new_img = torch.from_numpy(np.expand_dims(new_img, 0)).float()
    images_as_np.append(new_img)
#%%

transform = transforms.Compose([
    transforms.ToTensor()
])

#final dataframe is created
d = {'images': images_as_np, 'gray_matter': volume_list_ordered}
df = pd.DataFrame(d)

#getting names of each brain region for plotting purposes
#TODO: fix the names variables
names = df.iloc[0,1].index

def unnormalize(tensor: torch.Tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    Args:
        tensor: Tensor image of size (C, H, W) to be normalized.
        mean: Sequence of means for each channel.
        std: Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        torch.Tensor: Normalized Tensor image.

    Raises:
        TypeError: if `tensor` is not torch.Tensor
    """
    if not (torch.is_tensor(tensor) and tensor.ndimension() == 3):
        raise TypeError("tensor is not a torch image.")

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor

#model and data class
#data class
class FreeSurferData(Dataset):
  def __init__(self, df):
    self.image_list = df.iloc[:, 0]
    self.data_len = len(self.image_list)
    self.volume_list = df.iloc[:, -1]

  def __getitem__(self, index):
    #get one image
    im_normal = self.image_list[index]
    #get output
    label = volume_list[index].values.squeeze().astype('float32')
    return im_normal, label

  def __len__(self):
    return self.data_len


#CNN model
batch_size = 16
input_shape = [batch_size, 1, 256, 256, 256]

def conv_pool(*args, **kwargs):
    """Configurable Conv block with Batchnorm and Dropout"""
    return nn.Sequential(
        nn.Conv3d(*args, **kwargs),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1),
    )

params = [
    {
        "in_channels": 1,
        "kernel_size": 11,
        "out_channels": 144,
        "stride": 3,
    },
    {
        "in_channels": 144,
        "kernel_size": 5,
        "out_channels": 192,
        "stride": 2,
        "bias": False,
    },
    {
        "in_channels": 192,
        "kernel_size": 5,
        "out_channels": 192,
        "stride": 1,
        "bias": False,
    },


    ]

class model(nn.Module):
    """Configurable Net from https://www.frontiersin.org/articles/10.3389/fneur.2020.00244/full"""

    def __init__(self, n_classes):
        """Init"""

        super(model, self).__init__()
        layers = [conv_pool(**block_kwargs) for block_kwargs in params]
        layers.append(nn.Dropout3d(.4))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=5184000, out_features=374))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(.4))
        layers.append(nn.Linear(in_features=374, out_features=374))
        layers.append(nn.Linear(in_features=374, out_features=n_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class ModelWrapper(torch.nn.Module):
    def __init__(self, module, i):
        super().__init__()
        self.module = module
        self.i = i
    def forward(self, x):
        y = self.module(x)
        #getting last batch of data
        y = y[-1]
        #reshaping data so that the attribution function can use the data
        y = torch.reshape(y, (1,68))
        return y

n_classes = 68
#creating splits for test and train data and kfold

def attribute_image_features(algorithm, model, features, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(
        features,
        **kwargs
    )
    
    return tensor_attributions

features = df.iloc[0, 0]
print("Features shape: {}".format(features.shape))

features = torch.tensor(features).unsqueeze(0)
features.requires_grad = True


mymodel = model(n_classes)

logdir = "../logs/.../"
checkpoint = utils.load_checkpoint("/data/users2/vjoshi6/bin/pythonFiles/AIBM/logs100epochs/bmenet_gmve/train.100.pth")

utils.unpack_checkpoint(checkpoint, model=mymodel)
mymodel = ModelWrapper(mymodel, 0)

#difference = prediction.detach().numpy() - grad_label

#print("Difference: \n{}".format(difference))
#print("Shape of difference: {}".format(difference.shape))

ig = IntegratedGradients(mymodel)
saliency = Saliency(mymodel)
nt = NoiseTunnel(saliency)

print('CODE IS HERE')

image_name = str(single_image_path)

#TODO: change back to 67
#there are 68 classes; each run of the for loop plots attribution 
for i in range(0, 3):
    #NOTE: using target=i, but this might be incorrect
    mymodel.zero_grad()
    #getting attribution
    attr_ig = saliency.attribute(
        features,
        target=i)
    
    print("This is i: {}".format(i))

    """
    attr_ig, delta = ig.attribute(
        features,
        target=i,
        baselines=features * 0, 
        return_convergence_delta=True
    )
    """

    #getting the voxel value with the higest activation
    #attr_ig = torch.load('attr_ig.pt')
    arg_max = torch.argmax(attr_ig)

    #index of maximum activation for a given brain region
    idx = np.unravel_index(arg_max, attr_ig.shape)
    idx = (idx[1], idx[2], idx[3])

    print("Idx: {}".format(idx))
    torch.save(features, 'features_full_model_30_epoch.pt')
    torch.save(attr_ig, 'attr_ig_full_model_30_epoch.pt')

    #creating two arrays that can be modified to get slices
    attr_ig_viz = attr_ig.squeeze().cpu().detach().numpy()
    features_viz = features.squeeze().cpu().detach().numpy()
    
    print(names[i])
    if not os.path.isdir(names[i]):
        os.makedirs(names[i])

    # Organize the data for visualisation in the coronal plane
    coronal_feat = features_viz[idx[0], :, :]
    coronal_ig = attr_ig_viz[idx[0], :, :]
    # Organize the data for visualisation in the transversal plane
    transversal_feat = features_viz[:, idx[1], :]
    transversal_ig = attr_ig_viz[:, idx[1], :]
    # Organize the data for visualisation in the sagittal plane
    sagittal_feat = features_viz[:, :, idx[2]]
    sagittal_ig = attr_ig_viz[:, : , idx[2]]

    #transversal visualizations
    plt.imshow(transversal_feat)
    c = plt.imshow(transversal_ig, cmap='hot', alpha=0.4)
    plt.colorbar(c)
    plt.title(names[i])
    plt.savefig(names[i] + '/' + names[i] + 'imshowmethod_transversal_both.png')
    plt.clf()
    plt.imshow(transversal_ig, cmap='hot')
    plt.colorbar()
    plt.title(names[i])
    plt.savefig(names[i] + '/' + names[i] + 'imshowmethod_transversal_ig.png')
    plt.clf()

    #coronal visualizations
    plt.imshow(coronal_feat)
    c = plt.imshow(coronal_ig, cmap='hot', alpha=0.4)
    plt.colorbar(c)
    plt.title(names[i])
    plt.savefig(names[i] + '/' + names[i] + 'imshowmethod_coronal_both.png')
    plt.clf()
    plt.imshow(coronal_ig, cmap='hot')
    plt.colorbar()
    plt.title(names[i])
    plt.savefig(names[i] + '/' + names[i] + 'imshowmethod_coronal_ig.png')
    plt.clf()

    #saggittal visualizationns
    plt.imshow(sagittal_feat)
    c = plt.imshow(sagittal_ig, cmap='hot', alpha=0.4)
    plt.colorbar(c)
    plt.title(names[i])
    plt.savefig(names[i] + '/' + names[i] + 'imshowmethod_sagittal_both.png')
    plt.clf()
    plt.imshow(sagittal_ig, cmap='hot')
    plt.colorbar()
    plt.title(names[i])
    plt.savefig(names[i] + '/' + names[i] + 'imshowmethod_sagittal_ig.png')
    plt.clf()

    _ = viz.visualize_image_attr(
    attr_ig_viz, #(256,256,256) is shape and is a numpy array?
    features_viz, #(256,256,256) is shape and is a numpy array?
    method='blended_heat_map',
    cmap='hot',
    show_colorbar=True,
    sign='positive',
    outlier_perc=1
    )

    plt.title(names[i])
    plt.savefig(names[i] + '/' + names[i] + "Original_viz.png")
