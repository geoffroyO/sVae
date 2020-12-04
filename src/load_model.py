import cv2
import os
import skimage
import numpy as np
from tqdm import tqdm


def load_images(path_img, path_msk):
    spliced, copy_moved = [], []
    names = os.listdir(path_img)
    names.sort()
    spliced_msk, copy_moved_msk = [], []
    for name in tqdm(names):
        mask_name = name[:-4] + "_gt.png"
        msk = cv2.imread(path_msk + mask_name, 0)
        img = cv2.imread(path_img + name, 1)
        if msk is not None:
            msk = cv2.flip(msk, 1)
            Nx_msk, Ny_msk = msk.shape
            Nx_img, Ny_img, _ = img.shape
            if Nx_msk != Nx_img:
                img = cv2.transpose(img)
            Nx_msk, Ny_msk = msk.shape
            Nx_img, Ny_img, _ = img.shape
            if Nx_msk == Nx_img and Ny_msk == Ny_img:
                if name.split(".")[-1].lower() in {"jpeg", "jpg", "png", 'tif'}:
                    if name[3] == 'D':
                        spliced.append(img[..., ::-1])
                        spliced_msk.append(msk[..., ::-1])
                    else:
                        copy_moved.append(img[..., ::-1])
                        copy_moved_msk.append(msk[..., ::-1])
    return spliced, copy_moved, spliced_msk, copy_moved_msk


def patch_images(images, masks):
    data, labels = [], []
    for n, image in enumerate(tqdm(images)):
        patchs_img = extractPatches(image, (32, 32, 3), 8)
        patchs_msk = extractPatchesMask(masks[n], (32, 32), 8)
        for k, patch_img in enumerate(patchs_img):
            patch_msk = patchs_msk[k]
            name_img = "I_{}P_{}.jpg".format(n, k)
            name_msk = "I_{}P_{}_gt.jpg".format(n, k)
            # cv2.imwrite("../data/CASIA2_patch/Sp/{}".format(name_img), cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR))
            # cv2.imwrite("../data/CASIA2_patch/mask/{}".format(name_msk), patch_msk)
            data.append(patch_img)
            labels.append(patch_msk)
    return data, labels


def extractPatches(im, window_shape, stride):
    patches = skimage.util.view_as_windows(im, window_shape, stride)
    nR, nC, t, H, W, C = patches.shape
    nWindow = nR * nC
    patches = np.reshape(patches, (nWindow, H, W, C))

    return patches


def extractPatchesMask(msk, window_shape, stride):
    patches = skimage.util.view_as_windows(msk, window_shape, stride)
    nR, nC, H, W = patches.shape
    nWindow = nR * nC
    patches = np.reshape(patches, (nWindow, H, W))

    return patches

