import numpy as np
import re
from glob import glob
from os.path import basename

features, names, sids = [], [], []
pids = []

# Iterate through all the plates, and combine their features together
print("Start combining")
npz_list = glob("./output*/*/*.npz")
length = len(npz_list)

for i in range(length):
    print("Processed {:.2f}%".format(i / length * 100), end='\r')

    npz = npz_list[i]
    # Get the plate id
    name = basename(npz)
    pid = re.sub(r'(\d+)_.+_\d+\.npz', r'\1', name)
    sid = int(re.sub(r'\d+_.+_(\d+)\.npz', r'\1', name))

    if pid in ['24789', '25575', '26795']:
        print(pid)
        continue

    pids.append(pid)

    # Load the feature and modify the label
    feature = np.load(npz)['feature']
    features.append(feature)

    cpd = np.load(npz)['cpd']
    cpd = "{}_{}".format(pid, cpd)
    names.append(cpd)

    sids.append(sid)

# Save the combined features
features = np.vstack(features)
np.savez("combined_feature.npz", features=features, names=names)

print(len(set(pids)))
