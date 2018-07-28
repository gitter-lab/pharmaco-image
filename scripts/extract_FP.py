from rdkit import Chem
import deepchem as dc
import numpy as np
import pandas as pd


# Deepchem uses python2.7, so we add this separate script to extract
# fingerprints from the compounds.
def extract_fp(featurizer, smiles):
    """
    Extract fingerprint from the given smiles to a size long binary list.
    """
    molecule = Chem.MolFromSmiles(smiles)

    feature = featurizer._featurize(molecule).ToBitString()
    return list(map(int, list(feature)))


featurizer = dc.feat.CircularFingerprint(size=1024)

# Load all compounds in the dataset
bids, features = [], []
table = pd.read_csv("chemical_annotations_smiles.csv")
nan_counter = 0

for index, row in table.iterrows():
    # Extract feature
    bid = row['BROAD_ID']
    smiles = row['CPD_CANONICAL_SMILES']
    if pd.isna(smiles):
        nan_counter += 1
        continue
    print(bid, smiles)
    fp = extract_fp(featurizer, smiles)

    # Store feature with its compound name
    bids.append(bid)
    features.append(fp)

print("There are {} na's".format(nan_counter))
features = np.vstack(features)
np.savez('fp_features.npz', features=features, names=bids)
