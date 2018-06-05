from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np

df = pd.read_csv("./data/test/meta_data/chemical_annotations.csv")
canonical_smiles_list = []

for s in df['CPD_SMILES']:
    if pd.isna(s):
        canonical_smiles_list.append(np.nan)
        continue
    molecule = Chem.MolFromSmiles(s)
    canonical_smiles = Chem.MolToSmiles(molecule)
    canonical_smiles_list.append(canonical_smiles)

df['CPD_CANONICAL_SMILES'] = canonical_smiles_list
df.to_csv('./data/test/meta_data/chemical_annotations_smiles.csv', index=False)
