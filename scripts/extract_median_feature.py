import sqlite3
import pandas as pd
import numpy as np
from sys import argv
from json import load


def extract_cell_level_feature(c, table, tid, iid, filter_names,
                               validation_names):
    """
    Extract cell level features from the sql database.

    Args:
        c(sql_cursor): sqlite cursor
        table(string): table name, 'Cells', 'Cytoplasm', 'Nuclei'
        tid(string): TableNumber
        iid(int): ImageNumber(site number)
        filter_names([string]): features that are not extracted
        validation_names([string]): a list to validate the order of extracted
            features

    Return:
        np.array, median feature array
    """

    # Extract cell level features
    c.execute(
        """
        SELECT *
        FROM {}
        WHERE TableNumber = '{}' AND ImageNumber = {}
        """.format(table, tid, iid)
    )

    result = np.array(c.fetchall())

    # Filter out some features
    descriptions = [i[0] for i in c.description]
    droped_c = [i for i in range(len(descriptions)) if descriptions[i] in
                filter_names]
    result = np.delete(result, droped_c, axis=1)

    # Change the data type of result into floats
    result = result.astype(float)

    # Compute the medians of cell feature for this image
    median = np.median(result, axis=0)
    name = [i for i in descriptions if i not in filter_names]
    assert(name == validation_names)

    return median


def extract_feature(c, pid, wid, extract_info_dict):
    """
    Extract features from the well. Features are aggregated to a single image
    level. Therefore, if one well has 6 fields of view, there are 6 rows of
    the feature array.

    Args:
        c(sql_cursor): sqlite cursor
        pid(int): plate id
        wid(string): well id
        extract_info_dict(dict): a dictionary encoding extraction rules:
            extract_info_dict = {
                'image_feature_filter_name': [string]
                'image_name': [string]
                'cell_name': [string]
                'cytoplasm_name': [string]
                'nuclei_name': [string]
                'cell_feature_filter_name': [string]
                'cytoplasm_feature_filter_name': [string]
                'nuclei_feature_filter_name': [string]
            }

    Return:
        features: np.array(n, m): n fields of view with m features
        row_index: 'pid_wid_sid' string corresponding to each row of features
    """

    image_feature_filter_names = set(
        extract_info_dict['image_feature_filter_name']
    )

    # Extract all images (different sites / fields of view)
    c.execute(
        """
        SELECT TableNumber, ImageNumber
        FROM Image
        WHERE Image_Metadata_Plate = {} AND Image_Metadata_Well = '{}'
        """.format(pid, wid)
    )

    tid_iid_pairs = c.fetchall()

    # Track the feature names (row index)
    row_index = []
    features = []

    # Iterate through all sites and extract features from each site
    for p in tid_iid_pairs:
        tid, iid = p[0], p[1]

        # Extract image features
        c.execute(
            """
            SELECT *
            FROM Image
            WHERE TableNumber = '{}' AND ImageNumber = {}
            """.format(tid, iid)
        )

        result = c.fetchall()
        result = np.array(result[0])

        # Filter out some features
        descriptions = [i[0] for i in c.description]
        droped_c = [i for i in range(len(descriptions)) if descriptions[i] in
                    image_feature_filter_names]
        result = np.delete(result, droped_c, axis=0)

        # Change the data type of result into floats
        result = result.astype(float)

        image_feature = result
        image_name = [i for i in descriptions if i not in
                      image_feature_filter_names]
        assert(image_name == extract_info_dict['image_name'])

        # Extract cell, cytoplasm, and nuclei features
        cell_feature = extract_cell_level_feature(
            c,
            'Cells',
            tid,
            iid,
            set(extract_info_dict['cell_feature_filter_name']),
            extract_info_dict['cell_name']
        )

        cytoplasm_feature = extract_cell_level_feature(
            c,
            'Cytoplasm',
            tid,
            iid,
            set(extract_info_dict['cytoplasm_feature_filter_name']),
            extract_info_dict['cytoplasm_name']
        )

        nuclei_feature = extract_cell_level_feature(
            c,
            'Nuclei',
            tid,
            iid,
            set(extract_info_dict['nuclei_feature_filter_name']),
            extract_info_dict['nuclei_name']
        )

        # Combine image feature, cell level medians together
        cur_feature = np.hstack((image_feature,
                                 cell_feature,
                                 cytoplasm_feature,
                                 nuclei_feature))

        # Add the current feature into the well feature collections
        features.append(cur_feature)
        row_index.append('{}_{}_{}'.format(pid, wid, iid))

    features = np.vstack(features)
    return features, row_index


# Main function

# Load command line arguments
pid = int(argv[1])

# Load extraction configurations
extract_info_dict = load(open('./extract_info.json', 'r'))

# Load the mean table so we know the pid, wid, sid
df = pd.read_csv('./mean_well_profiles.csv')

# Connect the sql db
conn = sqlite3.connect('./{}.sqlite'.format(pid))
c = conn.cursor()

plate_features = []
plate_row_index = []
plate_row_bids = []

# Extract features well by well
for i, r in df.iterrows():
    if i == 3:
        break

    wid = r['Metadata_Well']

    features, row_index = extract_feature(c, pid, wid, extract_info_dict)
    plate_features.append(features)
    plate_row_index.extend(row_index)
    plate_row_bids.extend([r['Metadata_pert_id'] for i in
                           range(len(row_index))])

plate_features = np.vstack(plate_features)

# Output the features
np.savez('median_feature_{}.npz'.format(pid), features=plate_features,
         row_index=plate_row_index, bids=plate_row_bids)
