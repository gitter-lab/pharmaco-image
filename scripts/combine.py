import subprocess
import re
import cv2
import numpy as np
from math import ceil
from os import listdir, remove


def combine_images(prefix, row_size, img_format='png', input_dir='./',
                   output='./combined.png'):
    # Get the total number of images
    images = [i for i in listdir(input_dir) if
              re.search(r'^{}_.*\.{}'.format(prefix, img_format), i)]
    images.sort()

    # Combine images to multiple rows
    row_names = []
    temp_name = "tmp_row_{}.{}"

    # Make a white image for padding
    white = np.ones(cv2.imread(images[0]).shape) * 255
    cv2.imwrite("tmp_white.png", white)

    for i in range(ceil(len(images) / row_size)):
        upper = min((i + 1) * row_size, len(images))
        cur_images = images[i * row_size: (i + 1) * row_size]

        # Fill with white images if needed
        for _ in range((i + 1) * row_size - upper):
            cur_images.append("tmp_white.png")

        row_name = temp_name.format(i, img_format)
        row_names.append(row_name)
        subprocess.run(["convert"] + cur_images + ["+append", row_name])

    # Combine those rows
    subprocess.run(["convert"] + row_names + ["-append", output])

    # Remove temp files
    for t in row_names:
        remove(t)
    remove("tmp_white.png")


if __name__ == '__main__':
    combine_images("t-sne", 3, img_format='png')

