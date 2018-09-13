import cv2
import re
import numpy as np
import zipfile
import os
from shutil import copyfile, rmtree
from os.path import basename, join
from glob import glob

raw_channels = ['ERSyto', 'ERSytoBleed', 'Hoechst', 'Mito', 'Ph_golgi']
raw_paths = ['./{}-{}/*.tif'.format('{}', c) for c in raw_channels]


def merge_images(pid, wid, sid, g, out_dir='./'):
    """
    Merge images from 5 channels for all 6 snapshots.
    """
    print(pid, wid)
    # Load image names
    paths = [p.format(pid) for p in raw_paths]
    image_names, images = [], []
    for p in paths:
        print(p)
        cur_file = [f for f in glob(p) if
                    re.search(r'^.*_{}_s{}_.*\.tif$'.format(wid, sid),
                              basename(f))]
        # We should only see one result returned from the filter
        if len(cur_file) > 1:
            error = "Found more than one file for sid={} in {}"
            error = error.format(sid, p)
            raise ValueError(error)
        image_names.append(cur_file[0])

    # Read the images
    for n in image_names:
        images.append(cv2.imread(n, -1) * 16)

    # Need a dummy black image for merging
    black_image = np.zeros(images[0].shape).astype(images[0].dtype)

    # Save the merged image
    cv2.imwrite(join(out_dir, "{}_{}_s{}_45_{}.png".format(pid, wid, sid, g)),
                cv2.merge([images[4], black_image, images[3]]))
    cv2.imwrite(join(out_dir, "{}_{}_s{}_123_{}.png".format(pid, wid, sid, g)),
                cv2.merge([images[2], images[1], images[0]]))


if __name__ == '__main__':
    output = "./output"

    # Load the argument list for the plates to inspect
    tasks = []
    with open("args.txt", 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            words = line.replace('\n', '').split(',')
            pid = int(words[0])
            wid = words[1]
            sid = int(words[2])
            g = int(words[3])
            tasks.append((pid, wid, sid, g))

    # Sort the list of tuples by pid, so we only need to copy the same zip
    # file once
    tasks = sorted(tasks, key=lambda x: x[0])

    last_pid = 0
    for task in tasks:
        pid, wid, sid, g = task[0], task[1], task[2], task[3]

        if pid != last_pid:
            if last_pid != 0:
                # Clean the directory created by last pid
                for c in raw_channels:
                    rmtree("./{}-{}".format(last_pid, c))

            # Copy the zip file from Gluster to the current directory
            for c in raw_channels:
                copyfile("/mnt/gluster/zwang688/{}-{}.zip".format(pid, c),
                         "./{}-{}.zip".format(pid, c))

                # Extract the zip file and remove it
                with zipfile.ZipFile("./{}-{}.zip".format(pid, c), 'r') as fp:
                    fp.extractall('./')

                os.remove("./{}-{}.zip".format(pid, c))

            # Create an output dir for store the merged images
            os.mkdir(join(output, str(pid)))

        # Merge images
        merge_images(pid, wid, sid, g, out_dir=join(output, str(pid)))
        last_pid = pid

    for c in raw_channels:
        rmtree("./{}-{}".format(pid, c))
