"""Module for confounds extraction
"""
import numpy as np

from nilearn import _utils
from nilearn.image import high_variance_confounds, resample_img
from nilearn._utils.niimg_conversions import check_same_fov


def extract_confounds(imgs, mask_img, n_confounds=10):
    """Extract high variance confounds from one or more Nifti images.

    Parameters
    ----------
    imgs : Nifti1Image or list of Nifti1Image
        One or more functional 4D NIfTI images.

    mask_img : str or nibabel.Nifti1Image
        Binary gray matter mask.

    n_confounds : int
        Number of high-variance confound components to extract.

    Returns
    -------
    confounds : list of numpy.ndarray or single numpy.ndarray
        If one image is passed, returns a single array (n_timepoints, n_confounds).
        If multiple images are passed, returns a list of such arrays.
    """

    # Ensure imgs is a list of Nifti1Image objects
    if not isinstance(imgs, list):
        imgs = [imgs]

    confounds = []

    # Prepare for resampling if needed
    img0 = _utils.check_niimg_4d(imgs[0])
    shape = img0.shape[:3]
    affine = img0.affine

    if isinstance(mask_img, str):
        mask_img = _utils.check_niimg_3d(mask_img)

    if not check_same_fov(img0, mask_img):
        mask_img = resample_img(
            mask_img,
            target_shape=shape,
            target_affine=affine,
            interpolation='nearest',
            force_resample=True,
            copy_header=True
        )

    # Extract confounds from each image
    for img in imgs:
        print(f"[Confounds Extraction] Image selected: {img}")
        img = _utils.check_niimg_4d(img)
        print("Extracting high variance confounds")
        high_variance = high_variance_confounds(
            img, mask_img=mask_img, n_confounds=n_confounds
        )
        confounds.append(high_variance)

    return confounds[0] if len(confounds) == 1 else confounds

