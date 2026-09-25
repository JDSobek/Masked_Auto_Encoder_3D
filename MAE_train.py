"""
Script with a very simple training loop for a masked auto encoder to demonstrate how to use the model and ensure that the other code, and any modifications made, build a functional model.

Hyperparameters are generic values that are very probably suboptimal.
"""

from pathlib import Path
import sys

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from MAE_dataset import MAE_dataset
from MAE_model import MaskedAutoEncoder, functional_patchify


def main(train_data_folder, val_data_folder, save_path=None):

    device = 'cuda:0'
    num_workers = 2
    
    im_size = (224, 224, 224)
    patch_size = (16, 16, 16)

    num_epochs = 2
    batch_size = 2

    # running model with default settings for now
    model = MaskedAutoEncoder(img_size=im_size, patch_size=patch_size).to(device)

    train_dataset = MAE_dataset(train_data_folder, img_size=im_size)
    # using the same dataset for validation as training while prototyping
    val_dataset = MAE_dataset(val_data_folder, img_size=im_size)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.05)

    for i in range(num_epochs):
        # training loop
        print('train loop')
        model = model.train()
        for idx, (imgs, gt_img) in enumerate(train_dataloader):            
            imgs = imgs.to(device)
            gt_img = gt_img.to(device)
            labels = functional_patchify(gt_img, patch_size=patch_size)
            
            preds, mask = model(imgs)
            # remove classification token since we won't use it for the MAE training
            pred_patches = preds[:,1:,:]

            # only calculate loss on the masked patches
            # binary mask: 0 is keep, 1 is remove, inverting to mask via multiplication
            keep_patches = torch.ones_like(mask, device=device) - mask
            keep_patches = keep_patches.unsqueeze(-1)

            masked_labels = torch.mul(labels, keep_patches)
            masked_preds = torch.mul(pred_patches, keep_patches)

            optimizer.zero_grad()
            loss = criterion(masked_preds, masked_labels)
            loss.backward()
            optimizer.step()

            print(f'train epoch {i+1} batch loss: {loss.item()}')


        # validation loop
        print('val loop')
        with torch.no_grad():
            model = model.eval()
            for idx, (imgs, gt_img) in enumerate(val_dataloader):
                imgs = imgs.to(device)
                gt_img = gt_img.to(device)
                labels = functional_patchify(gt_img, patch_size=patch_size)

                preds, mask = model(imgs)
                # remove classification token since we won't use it for the MAE training
                pred_patches = preds[:,1:,:]

                # only calculate loss on the masked patches
                # binary mask: 0 is keep, 1 is remove, inverting to mask via multiplication
                keep_patches = torch.ones_like(mask, device=device) - mask
                keep_patches = keep_patches.unsqueeze(-1)

                masked_labels = torch.mul(labels, keep_patches)
                masked_preds = torch.mul(pred_patches, keep_patches)

                loss = criterion(masked_preds, masked_labels)
                print(f'val epoch {i+1} batch loss: {loss.item()}')

    # save final model
    if save_path:
        torch.save(model.state_dict(), save_path)

    return None


if __name__ == '__main__':
    import os
    img_folder = os.path.abspath(r"./TS_nifti_subset/")
    trial_savepath = os.path.abspath(r"./test_weights/mae_trial_run.pt")

    # the trial run uses the same dataset for training and validation because I was lazy
    main(train_data_folder=img_folder, val_data_folder=img_folder, save_path=trial_savepath)