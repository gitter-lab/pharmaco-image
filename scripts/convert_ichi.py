from rdkit import Chem
import pandas as pd
import numpy as np

df = pd.read_csv("./data/test/meta_data/chemical_annotations.csv")
canonical_smiles_list = []
inchi_list = []
errors = []

for s in df['CPD_SMILES']:
    if pd.isna(s):
        canonical_smiles_list.append(np.nan)
        inchi_list.append(np.nan)
        continue
    molecule = Chem.MolFromSmiles(s)
    canonical_smiles = Chem.MolToSmiles(molecule)
    canonical_smiles_list.append(canonical_smiles)
    inchi = Chem.MolToInchi(molecule)
    inchi_list.append(inchi)

df['CPD_CANONICAL_SMILES'] = canonical_smiles_list
df['INCHI'] = inchi_list
df.to_csv('./data/test/meta_data/chemical_annotations_inchi.csv', index=False)
