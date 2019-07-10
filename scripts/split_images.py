import deepchem as dc
import numpy as np
import pandas as pd
import re
from rdkit import Chem
from json import load, dump
from os.path import join, basename
from glob import glob


def scaffold_split(frac_train, frac_vali, frac_test):
    """
    Split compound indices into train, vali, and test set. The split is based
    on the compound scaffolds. The proportion is adjusted to the number of
    instances of each compound.

    Args:
        frac_train(float): proportion of the training set
        frac_vali(float): proportion of the validation set
        frac_test(float): proportion of the test set

    Returns:
        train_cmpd_index: set of compound indices (in the output matrix) in the
            training set
        vali_cmpd_index: set of compound indices in the validation set
        test_cmpd_index: set of compound indices in the test set
    """

    assert(frac_train + frac_vali + frac_test == 1)

    # Make a CSV file based on selected compounds and number of samples
    selected_well_dict = load(open('./selected_well_dict.json', 'r'))

    output_data = np.load('./output_matrix_convert_collision.npz',
                          allow_pickle=True)
    compound_inchi = output_data['compound_inchi']

    selected_cmpd_dict = {'cmpd_id': [],
                          'cmpd_activity': [],
                          'cmpd_smiles': []}

    # We will add multiple instances of one compound in this dataframe, so we
    # can use a dict to save time of translating inchi to smiles
    smiles_dict = {}

    for pid in selected_well_dict:
        for t in selected_well_dict[pid]:
            cur_cmpd_act, cur_cmpd_index = t[1], t[2]

            # Get smiles string
            if cur_cmpd_index in smiles_dict:
                cur_smiles = smiles_dict[cur_cmpd_index]
            else:
                cur_inchi = compound_inchi[cur_cmpd_index]
                mol = Chem.MolFromInchi(cur_inchi)
                cur_smiles = Chem.MolToSmiles(mol)

            # Populate the dict
            selected_cmpd_dict['cmpd_id'].append(cur_cmpd_index)
            selected_cmpd_dict['cmpd_activity'].append(cur_cmpd_act)
            selected_cmpd_dict['cmpd_smiles'].append(cur_smiles)

    cmpd_df = pd.DataFrame(selected_cmpd_dict)
    cmpd_df.to_csv('compound_df.csv', index=False)

    # Then, we create a deepchem dataset
    # We dont use this FP features tho
    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["cmpd_activity"]
    input_file = 'compound_df.csv'
    loader = dc.data.CSVLoader(tasks=tasks, smiles_field="cmpd_smiles",
                               featurizer=featurizer)
    dataset = loader.featurize(input_file)

    # Finally, use scatffold to split compounds
    splitter = dc.splits.splitters.ScaffoldSplitter(verbose=True)
    train_index, vali_index, test_index = splitter.split(dataset,
                                                         frac_train=frac_train,
                                                         frac_valid=frac_vali,
                                                         frac_test=frac_test)

    train_cmpd_index = set(cmpd_df.iloc[np.array(train_index), 1])
    vali_cmpd_index = set(cmpd_df.iloc[np.array(vali_index), 1])
    test_cmpd_index = set(cmpd_df.iloc[np.array(test_index), 1])

    # Make sure there is no compound overlapping between each two sets
    assert(len(train_cmpd_index.intersection(vali_cmpd_index)) == 0)
    assert(len(train_cmpd_index.intersection(test_cmpd_index)) == 0)
    assert(len(vali_cmpd_index.intersection(test_cmpd_index)) == 0)

    return train_cmpd_index, vali_cmpd_index, test_cmpd_index


# Generate train, vali, and test file list
def split_filenames(train_cmpd_index, vali_cmpd_index, test_cmpd_index,
                    img_dir, output_json='./scaffold_split.json'):
    """
    Split image tensors based on the given train, vali, and test compound
    indices.

    Args:
        train_cmpd_index(set[int]): list of compound indicies of training
            samples (file path)
        vali_cmpd_index(set[int]): list of compound indicies of validation
            samples (file path)
        test_cmpd_index(set[int]): list of compound indicies of test
            samples (file path)
        img_dir(str): directory containing all image tensors
        output_json(str): output json file
    """

    train_names, vali_names, test_names = [], [], []

    for f in glob(join(img_dir, '*.npz')):
        cur_cmpd_index = int(re.sub(r'img_\d+_.+_\d_\d_(\d+).npz', r'\1',
                                    basename(f)))
        if cur_cmpd_index in train_cmpd_index:
            train_names.append(f)
        elif cur_cmpd_index in vali_cmpd_index:
            vali_names.append(f)
        elif cur_cmpd_index in test_cmpd_index:
            test_names.append(f)
        else:
            print('Found unknown index {} from {}'.format(cur_cmpd_index, f))

    dump({'train_names': train_names,
          'vali_names': vali_names,
          'test_names': test_names}, open(output_json, 'w'), indent=2)


if __name__ == '__main__':
    # Get split compound indices
    frac_train = 0.6
    frac_vali = 0.2
    frac_test = 0.2
    train_cmpd_index, vali_cmpd_index, test_cmpd_index = scaffold_split(
        frac_train, frac_vali, frac_test
    )

    # Generate the file split json file
    split_filenames(train_cmpd_index, vali_cmpd_index, test_cmpd_index,
                    'images', output_json='./scaffold_split.json')
