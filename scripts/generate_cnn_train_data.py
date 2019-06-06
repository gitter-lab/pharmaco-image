import numpy as np
import re
import cv2
import os
import zipfile
from time import sleep
from sys import argv
from glob import glob
from os.path import join, basename
from shutil import copyfile, rmtree
from multiprocessing import Pool

# Global variables
raw_channels = ['ERSyto', 'ERSytoBleed', 'Hoechst', 'Mito', 'Ph_golgi']
raw_paths = ['./{}-{}/*.tif'.format('{}', c) for c in raw_channels]


def select_wells(assay):
    """
    Given an assay id, select pid and wid to extract images from.

    Args:
        assay(int): assay index (column of the output matrix)

    Return:
        a dictionary: {pid: [(wid, label), (wid, label)...]}
    """

    # Load the output matrix and each row's corresponding pid, wid
    output_data = np.load('./output_matrix_convert_collision.npz',
                          allow_pickle=True)
    output_matrix = output_data['output_matrix']
    pid_wids = output_data['pid_wids']

    # Find selected compounds in this assay
    selected_index = output_matrix[:, assay] != -1
    selected_labels = output_matrix[:, assay][selected_index]
    selected_pid_wids = np.array(pid_wids)[selected_index]

    # Flatten the selected pid_wids and group them by pid
    # selected_wells has structure [(wid, pid, label)]
    selected_wells = []

    for i in range(len(selected_pid_wids)):
        cur_pid_wids = selected_pid_wids[i]
        cur_label = selected_labels[i]

        for pid_wid in cur_pid_wids:
            selected_wells.append((pid_wid[0], pid_wid[1], int(cur_label)))

    # Group these wells by their pids
    selected_well_dict = {}
    for well in selected_wells:
        cur_pid, cur_wid, cur_label = well[0], well[1], well[2]

        if cur_pid in selected_well_dict:
            selected_well_dict[cur_pid].append((cur_wid, cur_label))
        else:
            selected_well_dict[cur_pid] = [(cur_wid, cur_label)]

    return selected_well_dict


def extract_instance(pid, wid, label, output_dir='./output'):
    """
    Extract all images in the given pid and wid pair (many fields of view).
    Then, it saves the extracted 5-channel images as a tensor in `output_dir`.

    Args:
        pid(int): plate id
        wid(int): well id
        label(int): 1 -> activated, 0 -> not activated
        output_dir(str): directory to store image tensors.
    """

    paths = [p.format(pid) for p in raw_paths]

    # Dynamically count number of sids for this pid-wid
    sid_files = [f for f in glob(paths[0]) if
                 re.search(r'^.*_{}_s\d_.*\.tif$'.format(wid), basename(f))]
    sid_num = len(sid_files)

    for sid in range(1, sid_num + 1):
        # Each sid generates one instance
        image_names, images = [], []

        for p in paths:
            # Search current pid-wid-sid
            cur_file = [f for f in glob(p) if
                        re.search(r'^.*_{}_s{}_.*\.tif$'.format(wid, sid),
                                  basename(f))]

            # We should only see one result returned from the filter
            if len(cur_file) != 1:
                error = "Found more than one file for {}-{}-{}.".format(
                    pid, wid, sid
                )
                raise ValueError(error)

            image_names.append(cur_file[0])

        # Read 5 channels
        for n in image_names:
            images.append(cv2.imread(n, -1) * 16)

        # Store each image as a 5 channel 3d matrix
        image_instance = np.array(images)

        # Save the instance with its label
        np.savez_compressed(join(output_dir, 'img_{}_{}_{}_{}.npz'.format(
            pid, wid, sid, label
        )), img=image_instance, )


def extract_plate(pid, selected_well_dict, output_dir='./output'):
    """
    Wrapper of `extract_instance()`. Copy zip files from gluster before
    extracting instances from one plate.

    Args:
        pid(int): plate id
        selected_well_dict(dict): {pid: [(wid, label), (wid, label)...]}
        output_dir(str): directory to store extracted tensors
    """

    print('Working on plate {} on process {}'.format(pid, os.getpid()))

    # Copy 5 zip files from gluster to the current directory
    for c in raw_channels:
        copyfile("/mnt/gluster/zwang688/{}-{}.zip".format(pid, c),
                 "./{}-{}.zip".format(pid, c))

        # Extract the zip file and remove it
        with zipfile.ZipFile("./{}-{}.zip".format(pid, c), 'r') as fp:
            fp.extractall('./')

        os.remove("./{}-{}.zip".format(pid, c))

    # Extract all instances from all selected wells in this plate
    for wid_tuple in selected_well_dict[pid]:
        extract_instance(pid, wid_tuple[0], wid_tuple[1], output_dir)

    # Clean up directories
    for c in raw_channels:
        rmtree("./{}-{}".format(pid, c))


if __name__ == '__main__':

    # Create the well dictionary using the command line argument
    assay = int(argv[1])
    nproc = int(argv[2])

    selected_well_dict = select_wells(assay)
    pids = selected_well_dict.keys()

    # Use multiprocessing to work on different plates at the same time
    # Prepare arguments for workers
    args = [(pid, selected_well_dict, './output') for pid in pids]

    # Feed args to nproc workers
    print("There are {} cpus on this node.".format(os.cpu_count()))
    pool = Pool(nproc)
    for arg in args:
        sleep(1)
        pool.apply_async(extract_plate, arg)
    pool.close()
    pool.join()
