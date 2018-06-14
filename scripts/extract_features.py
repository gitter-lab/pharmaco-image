import os
import re
import cv2
import time
import sqlite3
import argparse
import threading
import numpy as np
import pandas as pd
from shutil import rmtree
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from os.path import join, exists, basename
from glob import glob


#chemical_df = pd.read_csv("./data/test/meta_data/chemical_annotations" +
#                          "_smiles.csv")
chemical_df = pd.read_csv("./chemical_annotations_smiles.csv")
image_size = (696, 520)


class Plate():
    """
    A Plate object for one dataset (image + metadata).
    """
    def __init__(self, pid, sql_path, profile_path, input_dir):
        self.pid = pid
        self.input_dir = input_dir

        profile_df = pd.read_csv(profile_path)

        # Load the sql database
        conn = sqlite3.connect(sql_path)
        self.c = conn.cursor()

        # Load all well names and their associated sites of this plate
        self.c.execute(
            """
            SELECT Image_Metadata_Well, Image_Metadata_Site
            FROM Image
            WHERE Image_Metadata_Plate = {}
            """.format(pid)
        )

        results = self.c.fetchall()

        self.well_dict = {}
        for r in results:
            if r[0] in self.well_dict:
                self.well_dict[r[0]]['sid'].append(r[1])
            else:
                # Query the SMILES encoding of the compound used in this well
                broad_name = profile_df[profile_df['Metadata_Well'] == r[0]][
                    'Metadata_broad_sample'].values[0]
                if broad_name == "DMSO":
                    smiles = "DMSO"
                else:
                    smiles = chemical_df[chemical_df['BROAD_ID'] ==
                                         broad_name]['CPD_CANONICAL_SMILES']
                    smiles = smiles.values[0]

                # Store the data into the property dict
                self.well_dict[r[0]] = {
                    'cpd': smiles,
                    'sid': [r[1]]
                }

    def make_rgb_train_dirs(self, output_dir, nprocs, output_format='png'):
        """
        Generate a training directory having a subdirectory for each pid+wid.
        All associated channels and DOF are stored in those subdirectories.
        Cells are merged into RGB scale by 123 channels, and 45 channels.

        Args:
            nprocs: number of threads
            output_dir: the directory to save those subdirectories.
        """
        dies = {'ERSyto': 1,
                'ERSytoBleed': 2,
                'Hoechst': 3,
                'Mito': 4,
                'Ph_golgi': 5}

        if not exists(output_dir):
            os.mkdir(output_dir)

        lock = threading.Lock()
        threads = []
        lo = threading.local()
        job_queue = []
        well_dict = self.well_dict
        pid = self.pid
        input_dir = self.input_dir

        class Worker(threading.Thread):
            def __init__(self):
                super(Worker, self).__init__()

            def run(self):
                # Get the job
                while True:
                    lock.acquire()
                    if len(job_queue) == 0:
                        lock.release()
                        return
                    lo.wid = job_queue.pop()
                    lock.release()

                    # Make dir structure
                    lo.out_sub_dir = join(output_dir, '{}_{}'.format(pid,
                                                                     lo.wid))

                    if not exists(lo.out_sub_dir):
                        os.mkdir(lo.out_sub_dir)
                        os.mkdir(join(lo.out_sub_dir, 'c123'))
                        os.mkdir(join(lo.out_sub_dir, 'c45'))

                    for lo.sid in well_dict[lo.wid]['sid']:
                        lo.channels = []
                        for lo.d in dies:
                            lo.channel_dir = join(input_dir,
                                                  '{}-{}'.format(pid, lo.d))
                            lo.images = [f for f in os.listdir(lo.channel_dir)
                                         if re.search(r'^.*_{}_s{}_.*\.tif$'.
                                                      format(lo.wid, lo.sid),
                                                      f)]
                            lo.channels.append(cv2.imread(
                                join(lo.channel_dir, lo.images[0]), -1) * 16)

                        # Merge the channels and save to the output dir
                        lo.black_image = np.zeros(lo.channels[0].shape).astype(
                            lo.channels[0].dtype)
                        lo.c123_name = join(lo.out_sub_dir, 'c123/{}_{}_{}.{}'.
                                            format(pid, lo.wid, lo.sid,
                                                   output_format))
                        lo.c45_name = join(lo.out_sub_dir, 'c45/{}_{}_{}.{}'.
                                           format(pid, lo.wid, lo.sid,
                                                  output_format))
                        cv2.imwrite(lo.c45_name, cv2.merge([lo.channels[4],
                                                            lo.black_image,
                                                            lo.channels[3]]))
                        cv2.imwrite(lo.c123_name, cv2.merge([lo.channels[2],
                                                             lo.channels[1],
                                                             lo.channels[0]]))

        # Fill the job queue and launch workers
        job_queue = [wid for wid in well_dict]
        for t in range(nprocs):
            thread = Worker()
            thread.start()
            time.sleep(1)
            threads.append(thread)

        # Wait for all threads to finish
        for t in threads:
            t.join()

    def extract_features(self, image_dir, nprocs, image_format='png'):
        """
        Extract features from the merged rgb images using a pre-trained
        Inception v3 model.

        This method would run in multiple threads.

        Args:
            image_dir: the output directory of method self.make_rgb_train_dirs.
                The extracted features will also be stored in the sub-dir of
                this directory. Features are stored in npz format. Each well
                has # of sites npz files.
            nprocs: number of cpus in the device.
        """

        lock = threading.Lock()
        threads = []
        lo = threading.local()
        job_queue = []

        # Compile a bottleneck model for the threads
        base_model = InceptionV3(weights='imagenet', include_top=True)
        bottleneck_model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer("avg_pool").output
        )

        # Need this step for multi-threading
        # https://github.com/keras-team/keras/issues/6124
        bottleneck_model._make_predict_function()

        # Some global variables for threads to work
        black = [0, 0, 0]
        pad_size = int((image_size[0] - image_size[1]) / 2)
        v3_img_size = (299, 299, 3)
        pid = self.pid
        well_dict = self.well_dict

        # Build threads
        class FeatureExtractor(threading.Thread):
            def __init__(self, img_format):
                super(FeatureExtractor, self).__init__()
                self.img_format = img_format

            def run(self):
                """
                Extract feature from one sub-directory.
                """
                while True:
                    # Get the job
                    lock.acquire()
                    if len(job_queue) == 0:
                        lock.release()
                        return
                    print(len(job_queue))
                    lo.sub_dir = job_queue.pop()
                    lock.release()

                    # Get the wid
                    lo.wid = re.sub(r'^\d*_(.*)$', r'\1', basename(lo.sub_dir))
                    if well_dict[lo.wid]['cpd'] == "DMSO":
                        lo.wid = "DMSO"

                    for lo.img_name in glob(join(lo.sub_dir, 'c123/*.{}'.
                                                 format(self.img_format))):
                        lo.img1 = cv2.imread(lo.img_name, -1)
                        lo.img2 = cv2.imread(lo.img_name.replace('c123',
                                                                 'c45'), -1)

                        # Add padding to the images
                        lo.img1 = cv2.copyMakeBorder(lo.img1, pad_size,
                                                     pad_size, 0, 0,
                                                     cv2.BORDER_CONSTANT,
                                                     value=black)
                        lo.img2 = cv2.copyMakeBorder(lo.img2, pad_size,
                                                     pad_size, 0, 0,
                                                     cv2.BORDER_CONSTANT,
                                                     value=black)

                        # Resize the image to Inception v3 size
                        lo.img1 = cv2.resize(lo.img1, v3_img_size[:2])
                        lo.img2 = cv2.resize(lo.img2, v3_img_size[:2])
                        lo.batch = np.array([lo.img1, lo.img2])

                        # Get and save the features
                        lo.feature = np.squeeze(bottleneck_model.
                                                predict(lo.batch))
                        lo.feature = lo.feature.reshape(1, -1)
                        npz_name = basename(lo.img_name).replace('{}'.format(
                            self.img_format), 'npz')
                        np.savez(join(lo.sub_dir, npz_name),
                                 feature=lo.feature,
                                 cpd=lo.wid)

                    # Remove the image to save space after iterating all the
                    # images inside of those two dirs
                    if exists(join(lo.sub_dir, 'c123')):
                        rmtree(join(lo.sub_dir, 'c123'))
                    if exists(join(lo.sub_dir, 'c45')):
                        rmtree(join(lo.sub_dir, 'c45'))

        # Filling the job queue
        job_queue = [f.path for f in os.scandir(image_dir) if f.is_dir() and
                     str(pid) in f.path]

        # Start the jobs
        threads = []
        for t in range(nprocs):
            thread = FeatureExtractor(image_format)
            thread.start()
            time.sleep(1)
            threads.append(thread)

        # Wait for all threads to finish
        for t in threads:
            t.join()


if __name__ == '__main__':
    # Add cli
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", help="the plate id", type=int)
    parser.add_argument("input_dir", help="the input directory path")
    parser.add_argument("output_dir", help="the path of output directory")
    parser.add_argument("sql_path", help="the path of sql metadata database")
    parser.add_argument("profile_path", help="the path of mean profile csv")
    parser.add_argument("nprocs", help="number of threads", type=int)
    args = parser.parse_args()

    plate = Plate(args.pid,
                  args.sql_path,
                  args.profile_path,
                  args.input_dir)
    plate.make_rgb_train_dirs(args.output_dir, args.nprocs)
    plate.extract_features(args.output_dir, args.nprocs)

    """
    pid = 24278
    sql_path = './data/test/meta_data/extracted_features/24278.sqlite'
    input_dir = '/Users/JayWong/Downloads'
    profile_path = './data/test/meta_data/profiles/mean_well_profiles.csv'
    output_dir = './test'
    nprocs = 4

    plate = Plate(pid,
                  sql_path,
                  profile_path,
                  input_dir)
    #plate.make_rgb_train_dirs(output_dir, nprocs)
    plate.extract_features(output_dir, nprocs)
    """
