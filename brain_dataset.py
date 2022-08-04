from typing import Any, Callable, List, Union
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset

#import torchio as tio

#import nobrainer

_Path = Union[str, Path]


class BrainDataset(Dataset):
    """General purpose dataset class with several data sources `list_data`."""

    def __init__(self,
                 df: pd.DataFrame,
                 open_fn: Callable,
                 input_key: str = "images"
                ):
        """
        Args:
            list_data (List[Dict]): list of dicts, that stores
                you data annotations,
                (for example path to images, labels, bboxes, etc.)
            open_fn (callable): function, that can open your
                annotations dict and
                transfer it to data, needed by your network
                (for example open image by path, or tokenize read string.)
        """
        self.data = df
        self.open_fn = open_fn

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.data)

    def __getitem__(self, index: Union[int, list]) -> Any:
        """Gets element of the dataset.
        Args:
            index (int): index of the element in the dataset
        Returns:
            List of elements by index
        """
        data_dict_list = self.data.to_dict('records')
        image = self.open_fn(data_dict_list[index])
        labels = np.array([
            value for key, value in data_dict_list[0].items() if 'volume_paths' not in key])
        image = torch.from_numpy(np.expand_dims(image, 0)).float()
        labels = torch.from_numpy(labels).float()
        #transform = tio.RandomAffine(
        #    degrees=90,
        #    translation=15
        #)
        #image = transform(image)
        #nobrainer.volume.apply_random_transform_scalar_labels(image, labels)

        return image, labels
