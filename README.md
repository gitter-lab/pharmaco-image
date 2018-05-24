# pharmaco-image

## Read the Tiff Image

The best way is to use `skim age.external.tifffile` to handle the Tiff image IO. This module is basically a wrapper of the [`tifffile.py`](https://pypi.org/project/tifffile/), but it comes with `scikit-image` and better documentation.

To display the read image data, simply use `matplotlib`'s `imshow` function.