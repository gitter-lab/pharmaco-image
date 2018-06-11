import os
import re
import cv2
import time
import sqlite3
import threading
import pandas as pd
import numpy as np
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from os.path import join, exists, basename
from glob import glob


chemical_df = pd.read_csv("./data/test/meta_data/chemical_annotations" +
                          "_smiles.csv")
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

    def make_rgb_train_dirs(self, output_dir, output_format='png'):
        """
        Generate a training directory having a subdirectory for each pid+wid.
        All associated channels and DOF are stored in those subdirectories.
        Cells are merged into RGB scale by 123 channels, and 45 channels.

        Args:
            output_dir: the directory to save those subdirectories.
        """
        dies = {'ERSyto': 1,
                'ERSytoBleed': 2,
                'Hoechst': 3,
                'Mito': 4,
                'Ph_golgi': 5}

        if not exists(output_dir):
            os.mkdir(output_dir)

        for wid in self.well_dict:
            # Make dir structure
            out_sub_dir = join(output_dir, '{}_{}'.format(self.pid, wid))

            if not exists(out_sub_dir):
                os.mkdir(out_sub_dir)
                os.mkdir(join(out_sub_dir, 'c123'))
                os.mkdir(join(out_sub_dir, 'c45'))

            for sid in self.well_dict[wid]['sid']:
                channels = []
                for d in dies:
                    channel_dir = join(self.input_dir, '{}-{}'.format(self.pid,
                                                                      d))
                    images = [f for f in os.listdir(channel_dir) if
                              re.search(r'^.*_{}_s{}_.*\.tif$'.format(
                                  wid, sid), f)]
                    channels.append(cv2.imread(join(channel_dir,
                                                    images[0]), -1) * 16)

                # Merge the channels and save to the output dir
                black_image = np.zeros(channels[0].shape).astype(
                    channels[0].dtype)
                c123_name = join(out_sub_dir, 'c123/{}_{}_{}.{}'.format(
                    self.pid, wid, sid, output_format))
                c45_name = join(out_sub_dir, 'c45/{}_{}_{}.{}'.format(
                    self.pid, wid, sid, output_format))
                cv2.imwrite(c45_name, cv2.merge([channels[4],
                                                 black_image,
                                                 channels[3]]))
                cv2.imwrite(c123_name, cv2.merge([channels[2],
                                                  channels[1],
                                                  channels[0]]))

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
                    lo.sub_dir = job_queue.pop()
                    lock.release()

                    for lo.img_name in glob(join(lo.sub_dir, 'c123/*.{}'.\
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
                                 feature=lo.feature)


        # Filling the job queue
        job_queue = [f.path for f in os.scandir(image_dir) if f.is_dir()]

        # Start the jobs
        threads = []
        for t in range(4):
            thread = FeatureExtractor(image_format)
            thread.start()
            time.sleep(1)
            threads.append(thread)

        # Wait for all threads to finish
        for t in threads:
            t.join()


if __name__ == '__main__':
    sql_path = './data/test/meta_data/extracted_features/24278.sqlite'
    input_dir = '/Users/JayWong/Downloads'
    profile_path = './data/test/meta_data/profiles/mean_well_profiles.csv'
    plate = Plate(24278, sql_path, profile_path, input_dir)
    # plate.make_rgb_train_dirs('./test')
    plate.extract_features('./test', 1)
