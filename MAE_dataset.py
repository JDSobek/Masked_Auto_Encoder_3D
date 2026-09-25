"""
A dataset that reads in 3-D NIfTI files for use with a 3-D ViT masked auto encoder.

This is mostly just repurposed MedYOLO code.
"""

import os
from typing import Tuple, List

import nibabel as nib
import numpy as np

import torch


def load_nifti(filepath: str) -> Tuple[torch.Tensor, np.array]:
    """Read a NIfTI then convert the data into a torch tensor,
    and copy the affine data into a numpy array for downstream reconstruction.

    Args:
        filepath (str): Path to the NIfTI file

    Returns:
        nifti (torch.tensor): tensor containing the unnormalized NIfTI image data
        nifti_affine (np.array): array containing the NIfTI affine data
    """
    nifti = nib.load(filepath)
    nifti_affine = nifti.affine
    nifti_array = np.array(nifti.dataobj)
    assert nifti_array is not None, 'Image Not Found ' + filepath    
    nifti_tensor = torch.tensor(nifti_array, dtype=torch.float)
    return nifti_tensor, nifti_affine


def transpose_nifti_shape(nifti_tensor: torch.Tensor) -> torch.Tensor:
    """Transposes the NIfTI tensor dimensions from height, width, depth order to depth, height, width
    to make it compatible with torch convolutions.

    Args:
        nifti_tensor (torch.tensor): tensor to reshape

    Returns:
        nifti_tensor (torch.tensor): reshaped tensor
    """
    nifti_tensor = torch.transpose(nifti_tensor, 0, 2)
    nifti_tensor = torch.transpose(nifti_tensor, 1, 2)
    return nifti_tensor


def reshape_input(nifti_tensor: torch.Tensor, new_size: Tuple[int]) -> torch.Tensor:
    """Reshapes a 3D tensor to have new_size and an added channel dimension.

    Args:
        nifti_tensor (torch.Tensor): The tensor to be reshaped
        new_size (int): The new edge_length

    Returns:
        nifti_tensor (torch.tensor): Resized tensor
    """
    # add channel dimension for compatibility with later code
    nifti_tensor = torch.unsqueeze(nifti_tensor, 0)
    # add batch dimension for functional interpolate
    nifti_tensor = torch.unsqueeze(nifti_tensor, 0)
    # resize image to a cube of size new_size
    nifti_tensor = torch.nn.functional.interpolate(nifti_tensor, size=new_size, mode='trilinear', align_corners=False)
    # remove batch dimension for compatibility with later code
    nifti_tensor = torch.squeeze(nifti_tensor, 0)
    return nifti_tensor


def normalize_CT(imgs: torch.Tensor) -> torch.Tensor:
    """Normalizes 3D CTs in Hounsfield Units (+/- 1024) to within 0 and 1.

    Args:
        imgs (torch.tensor): unnormalized model input

    Returns:
        imgs (torch.tensor): normalized model input
    """
    imgs = (imgs + 1024.) / 2048.0  # int to float32, -1024-1024 to 0.0-1.0
    return imgs


def find_niftis(data_folder: str) -> List[str]:
    """
    Extract NIfTI files from a directory to create a list of input paths for the dataset.

    Args:
        data_folder (str): path to the folder containing the input NIfTIs
    
    Returns:
        imagefile_list (list): list of the paths to the input NIfTIs
    """
    imagefile_list = []
    for dirpath, subdirs, files in os.walk(data_folder):
            for file in files:
                # can only use NIfTIs, so filter out any other files in the folder
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    file_path = os.path.join(dirpath, file)
                    imagefile_list.append(file_path)
    return imagefile_list


class MAE_dataset(torch.utils.data.Dataset):
    """
    Dataset of NIfTI inputs for use with the masked autoencoder.
    """
    def __init__(self, data_folder: str, img_size: Tuple[int, int, int]):
        super().__init__()
        self.data_folder = data_folder
        self.img_size = img_size
        self.image_filelist = find_niftis(self.data_folder)

    def __len__(self):
        return len(self.image_filelist)
    
    def __getitem__(self, index):
        # load the NIfTI input
        image_path = self.image_filelist[index]
        nifti_tensor, _ = load_nifti(image_path)

        # preprocess input
        nifti_tensor = transpose_nifti_shape(nifti_tensor)
        nifti_tensor = reshape_input(nifti_tensor, self.img_size)
        nifti_tensor = normalize_CT(nifti_tensor)

        # remove gradient from the label
        nifti_label = nifti_tensor.detach().clone()
        return nifti_tensor, nifti_label
    

if __name__ == '__main__':
    # test folder with the NIfTIs
    data_folder = os.path.abspath(r"./TS_nifti_subset/")

    img_size = (224, 224, 224)
    dataset = MAE_dataset(data_folder=data_folder, img_size=img_size)

    # testing the dataset works
    for i in range(len(dataset)):
        tensor_in, label = dataset.__getitem__(i)
        print('--------------')
        print(tensor_in.size())
        print(label.size())