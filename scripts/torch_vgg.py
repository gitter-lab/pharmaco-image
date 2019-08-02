import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from glob import glob
from os.path import join, basename
from json import dump, load
from sklearn.utils import shuffle
from sklearn import metrics
from sys import argv
from collections import Counter


class Dataset(data.Dataset):
    """
    Define a dataset class so we can load data in the runtime.
    Trainning dataset, vali dataset and test dataset can all use this class.
    """

    def __init__(self, img_names):
        """
        Args:
            img_names([string]): a list of image names in this dataset. The
                name should be a relative path to a single image.
        """

        self.img_names = img_names

    def __len__(self):
        """
        Tell pytorch how many instances are in this dataset.
        """

        return len(self.img_names)

    def __getitem__(self, index):
        """
        Generate one image instance based on the given index.

        Args:
            index(int): the index of the current item

        Return:
            x(tensor): 5-channel 3d tensor encoding one cell image
            y(int): 0 - negative assay, 1 - positive assay
        """

        # Read the image matrix and convert to torch tensor
        cur_img_name = self.img_names[index]
        mat = np.load(cur_img_name)['img'].astype(dtype=np.float32)
        x = torch.from_numpy(mat)

        # Get the image label from its filename
        y = int(re.sub(r'img_\d+_.+_\d_(\d)_\d+\.npz', r'\1',
                       basename(cur_img_name)))

        return x, y


class VGG(nn.Module):
    """
    Modified VGG 11 architecture.
    """

    def __init__(self):
        """
        Create layers for the LeNet network.
        """

        super(VGG, self).__init__()

        layers = []
        layers.append(nn.Conv2d(5, 32, kernel_size=3, padding=1))
        layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 200),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(200, 200),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(200, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Pytorch forward() method for autogradient.
        """

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x


def train_one_epoch(model, device, training_generator, vali_generator,
                    optimizer, criterion, epoch, early_stopping=None):
    """
    Function to train the CNN for one epoch.

    Args:
        model(torch model): CNN model
        device(torch device): cpu or gpu
        training_generator(torch data generator): training set data generator
        vali_generator(torch data generator): vali set data generator
        optimizer(torch optimizer): optimizer
        criterion(torch criterion): loss function
        epoch(int): epoch index
        early_stopping(dict): early stopping dictionary to track best loss
            value on the validation set and how many epochs since last
            improvement

    Returns:
        train_loss: loss value on the training set
        train_acc: accuracy on the training set
        vali_loss: loss value on the validation set
    """

    # Set lenet to training mode
    model.train()

    train_losses, y_predict_prob, y_true = [], [], []
    for i, (cur_batch, cur_labels) in enumerate(training_generator):

        cur_labels_array = [i.item() for i in cur_labels]
        print(i, Counter(cur_labels_array))

        # Transfer tensor to GPU if available
        cur_labels = cur_labels.float()
        cur_batch, cur_labels = cur_batch.to(device), cur_labels.to(device)

        # Clean the gradient
        optimizer.zero_grad()

        # Run the network forward
        output = torch.squeeze(model(cur_batch))

        # Compute the loss
        loss = criterion(output, cur_labels)
        train_losses.append(loss.detach().item())
        y_predict_prob.extend(output.cpu().detach().numpy())
        y_true.extend(cur_labels.cpu().numpy())

        # Backpropogation and update weights
        loss.backward()
        optimizer.step()

    # Convert tensor to numpy array so we can use sklearn's metrics
    y_predict_prob = np.stack(y_predict_prob)
    y_predict = [1 if p >= 0.5 else 0 for p in y_predict_prob]
    y_true = np.array(y_true)

    # Average losses over different batches. Each loss corresponds to the mean
    # loss within that batch (reduction="mean").
    train_loss = np.mean(train_losses)
    train_acc = metrics.accuracy_score(y_true, y_predict)

    # After training for this epoch, we evaluate this current model on the
    # validation set
    model.eval()
    vali_losses = []

    with torch.no_grad():
        for cur_batch, cur_labels in vali_generator:
            cur_labels = cur_labels.float()
            cur_batch, cur_labels = cur_batch.to(device), cur_labels.to(device)
            output = torch.squeeze(model(cur_batch))

            loss = criterion(output, cur_labels)
            vali_losses.append(loss.detach().item())

    # Average losses over different batches. Each loss corresponds to the mean
    # loss within that batch (reduction="mean").
    vali_loss = np.mean(vali_losses)

    # Early stopping (the real stopping is outside of this function)
    if early_stopping:
        if vali_loss < early_stopping['best_loss']:
            early_stopping['best_loss'] = vali_loss
            early_stopping['wait'] = 0
        else:
            early_stopping['wait'] += 1

    return train_loss, train_acc, vali_loss


def test(model, device, criterion, test_generator):
    """
    Evaluate the model on test dataset.

    Args:
        model(torch model): trained model
        device(torch device): cpu/gpu
        criterion(torch criterion): loss function
        test_generator(torch datagenerator): test dataset generator

    Returns:
        test_loss: loss on the test set
        test_acc: accuracy on the test set
        y_predict_prob: predicted positive score
        y_true: true label
    """

    # Set model to evaluation mode
    model.eval()

    test_losses, y_predict_prob, y_true = [], [], []

    with torch.no_grad():
        for cur_batch, cur_labels in test_generator:
            # Even there is only forward() in testing phase, it is still faster
            # to do it on GPU
            cur_labels = cur_labels.float()
            cur_batch, cur_labels = cur_batch.to(device), cur_labels.to(device)

            output = torch.squeeze(model(cur_batch))
            loss = criterion(output, cur_labels)

            # Track the loss and prediction for each batch
            test_losses.append(loss.detach().item())
            y_predict_prob.extend(output.cpu().detach().numpy())
            y_true.extend(cur_labels.cpu().numpy())

    # Convert tensor to numpy array so we can use sklearn's metrics
    y_predict_prob = np.stack(y_predict_prob)
    y_predict = [1 if i >= 0.5 else 0 for i in y_predict_prob]

    # Need to cast np.int64 to int for json dump
    y_true = list(map(int, y_true))

    # Take the average of batch loss means
    test_loss = np.mean(test_losses)
    test_acc = metrics.accuracy_score(y_true, y_predict)

    print("Testing on {} instances, the accuracy is {:.2f}.".format(
        len(test_generator), test_acc
    ))

    return test_loss, test_acc, y_predict_prob, y_true


def generate_data(bs, nproc, img_dir='./images', split_json=None):
    """
    Create generators for train, vali, and test dataset. If `split_json`
    is given, it will use the pre-defined dataset split rules. Otherwise, it
    uses a random 6:2:2 split.

    Args:
        bs(int): batch size
        nproc(int): how many subprocesses to use for data loading
        img_dir(str): directory containing all image tensors. This is not used
            if `cv_split_json` is given.
        split_json(str): filename of the json file where each image tensor is
            grouped under either train, vali, or test set.

    Returns:
        Three data generators corresponding to train, vali, and test dataset.
        weights(1d tensor): class weights computed by the label count in the
            training data.
    """

    # Randomly split img_names into three sets by 6:2:2
    if not split_json:
        img_names = glob(join(img_dir, '*.npz'))

        img_names = shuffle(img_names)
        quintile_len = len(img_names) // 5
        vali_names = img_names[: quintile_len]
        test_names = img_names[quintile_len: quintile_len * 2]
        train_names = img_names[quintile_len * 2:]

    # Or, use pre-defined train, vali, and test dataset
    else:
        split_dict = load(open(split_json, 'r'))
        train_names = split_dict['train_names']
        vali_names = split_dict['vali_names']
        test_names = split_dict['test_names']

    print(("There are {} training samples, {} validation samples, " +
           "and {} test samples.").format(
               len(train_names), len(vali_names), len(test_names)))

    """
    # Count class labels in the training set to assign class weights
    training_labels = []
    for n in train_names:
        label = int(re.sub(r'img_\d+_.+_\d_(\d)_\d+\.npz', r'\1', basename(n)))
        training_labels.append(label)

    label_count = Counter(training_labels)
    print('Training label counter: {}'.format(label_count))
    weights = [1, 1]
    if label_count[0] > label_count[1]:
        weights[1] = label_count[0] / label_count[1]
    else:
        weights[0] = label_count[1] / label_count[0]

    # Convert class weights to a 1D tensor
    weights = torch.from_numpy(np.array(weights)).float()
    """

    # Count sample weight in the training set so we can sample balanced data
    # in each batch later
    training_labels = []
    for n in train_names:
        label = int(re.sub(r'img_\d+_.+_\d_(\d)_\d+\.npz', r'\1', basename(n)))
        training_labels.append(label)

    label_count = Counter(training_labels)
    label_count_array = torch.tensor([label_count[0], label_count[1]],
                                     dtype=torch.float)

    # The class weight here is the probability to sample this instance given
    # its class. Therefore, underrepresented class should have a higher
    # class weight (probability).
    class_weights = 1.0 / label_count_array

    # Then assign the weight to each sample
    sample_weights = [class_weights[l].item() for l in training_labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    # Create a random sample sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    # batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=bs,
    #                                               drop_last=False)
    print('Training label counter: {}'.format(label_count))

    # Create data generators
    train_params = {
        'batch_size': bs,
        'shuffle': False,
        'sampler': sampler,
        'num_workers': nproc
    }

    vali_or_test_params = {
        'batch_size': bs,
        'shuffle': True,
        'num_workers': nproc
    }

    training_dataset = Dataset(train_names)
    training_generator = data.DataLoader(training_dataset, **train_params)

    vali_dataset = Dataset(vali_names)
    vali_generator = data.DataLoader(vali_dataset, **vali_or_test_params)

    test_dataset = Dataset(test_names)
    test_generator = data.DataLoader(test_dataset, **vali_or_test_params)

    return training_generator, vali_generator, test_generator


def train_main(assay, lr, bs, nproc, patience, epoch, img_dir='./images'):
    """
    Main function to train a LeNet CNN on these image tensors. This function
    generates a json file encoding all training information and test results.

    Args:
        assay(int): assay index
        lr(float): learning rate
        bs(int): batch size
        nproc(int): nubmer of workers
        patience(int): early stopping patience
        epoch(int): max epoch
        img_dir(str): directory containing all the image tensors
    """

    # Generate three datasets
    (training_generator, vali_generator,
     test_generator) = generate_data(bs, nproc, img_dir=img_dir,
                                     split_json='scaffold_split.json')

    # Run on GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on {}.".format(device))

    # Use cross-entropy as our loss funciton
    model = VGG()
    model = model.to(device)

    # Need to transfer weight tensor to GPU
    # weights = weights.to(device)
    # criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean')
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion = nn.BCELoss(reduction='mean')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Init early stopping config
    early_stopping_dict = {
        'best_loss': np.inf,
        'wait': 0,
        'patience': patience
    }

    train_losses, train_accs, vali_losses = [], [], []

    for e in range(epoch):
        # Train one epoch
        train_loss, train_acc, vali_loss = train_one_epoch(
            model,
            device,
            training_generator,
            vali_generator,
            optimizer,
            criterion,
            e,
            early_stopping=early_stopping_dict
        )

        # Track training process
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        vali_losses.append(vali_loss)

        # Early stopping
        if early_stopping_dict['wait'] > early_stopping_dict['patience']:
            print('Early stop: best vali loss = {}, waited {} epochs.'.format(
                early_stopping_dict['best_loss'],
                early_stopping_dict['wait']))
            break

    # After training, evaluate trained model on the test set
    test_loss, test_acc, y_predict_prob, y_true = test(
        model, device, criterion, test_generator
    )

    # Save weights, results
    torch.save(model.state_dict(), './trained_weights_{}_{}_{}.pt'.format(
        assay, lr, bs
    ))

    results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'vali_losses': vali_losses,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'y_predict_prob': y_predict_prob.astype(float).tolist(),
        'y_true': y_true
    }

    dump(results, open('./results_{}_{}_{}.json'.format(assay, lr, bs), 'w'),
         indent=2)


if __name__ == '__main__':

    # Parse command line arguments
    assay, bs, lr, nproc = (int(argv[1]),
                            int(argv[2]),
                            float(argv[3]),
                            int(argv[4]))
    epoch = 300
    patience = 30
    img_dir = './images'
    train_main(assay, lr, bs, nproc, patience, epoch, img_dir=img_dir)

