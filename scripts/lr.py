import sys
import numpy as np
from collections import Counter
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import SGDClassifier


def train_on_assay(x, y, aid, lr, kfold=5, n_jobs=4):
    """
    Train and measure a logistic regression model in a cross validation
    fashion.

    The model is only trained on compounds which give non-NA results.
    """

    if Counter(y)[1] < kfold:
        print("Warning: the number of total postives is less than kfold")
        return

    # Check if the training samples are too small
    if Counter(y)[1] < 10 or Counter(y)[0] < 10:
        print("Not enough training samples for assay {}".formt(aid))
        return

    # Build the cross validation scheme
    # Since some assays have extremely small number of positives,
    # I will use StratifiedKFold to preserve the proportion of positive
    # in test set

    my_kfold_gen = StratifiedKFold(n_splits=kfold, shuffle=True)
    scoring = ["f1", "accuracy", "precision", "recall", "average_precision",
               "roc_auc"]
    cv = cross_validate(lr, x, y, scoring=scoring, cv=my_kfold_gen,
                        n_jobs=n_jobs, verbose=1,
                        return_train_score=False)

    cv["total_count"] = Counter(y)

    return cv


# Parse the commands: col max_iter solver
i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 100
solver = sys.argv[3] if len(sys.argv) > 3 else "liblinear"

if not i:
    print("Please provide commandline arguments for i, max_iter, solver")
    exit()

print("Start training, assay={}, max_iter={}, solver={}".format(
    i, max_iter, solver))

lr_classifier = SGDClassifier(penalty='l1', class_weight="balanced",
                              max_iter=max_iter, verbose=0)

# feature_index = np.load("feature_index.npz")["index"]

output_matrix = np.load("./output_matrix_collision_inception.npz")[
    "output_matrix"]
features = np.load("./matched_collision_raw_features.npz")["features"]

# They key is to clean the memory right after sub-selecting the data
assay_array = output_matrix[:, i].copy()
output_matrix = None

y_index = [False if a == -1 else True for a in assay_array]
y = assay_array[y_index].copy()
assay_array = None

x = features[y_index, :].copy()
# x = x[:, feature_index].copy()
features = None
y_index = None

result = train_on_assay(x, y, i,
                        lr_classifier,
                        kfold=5,
                        n_jobs=4)

np.savez("results_assay_{}.npz".format(i), result=result)
