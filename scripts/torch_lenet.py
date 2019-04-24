import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from glob import glob
from os.path import join, basename
from json import dump
from sklearn.utils import shuffle
from sklearn import metrics
from sys import argv


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
        y = int(re.sub(r'img_\d+_.+_\d_(\d)\.npz', r'\1',
                       basename(cur_img_name)))

        return x, y


class LeNet(nn.Module):
    """
    Modified LeNet architecture.
    """

    def __init__(self):
        """
        Create layers for the LeNet network.
        """

        super(LeNet, self).__init__()

        # C1: 5 channel -> 6 filters (5x5)
        self.conv1 = nn.Conv2d(5, 6, 5)
        # C2: 6 filters -> 16 filters (5x5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # FC1: CP2 -> 120
        self.fc1 = nn.Linear(171 * 171 * 16, 120)
        # FC2: FC1 -> 84
        self.fc2 = nn.Linear(120, 84)
        # Output: FC2 -> 2 (activated or not)
        self.output = nn.Linear(84, 2)

    def forward(self, x):
        """
        Pytorch forward() method for autogradient.
        """

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)

        # Flatten this layer to connect to FC lyaers
        # size(0) is the batch size
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # The original lenet5 doesnt use softmax.
        out = F.softmax(self.output(out), dim=1)

        return out


def train_one_epoch(model, device, training_generator, vali_generator,
                    optimizer, criterion, epoch, early_stopping=None):

    # Set lenet to training mode
    model.train()

    train_losses, y_predict_prob, y_true = [], [], []
    for i, (cur_batch, cur_labels) in enumerate(training_generator):

        # Transfer tensor to GPU if available
        cur_batch, cur_labels = cur_batch.to(device), cur_labels.to(device)

        # Clean the gradient
        optimizer.zero_grad()

        # Run the network forward
        output = model(cur_batch)

        # Compute the loss
        loss = criterion(output, cur_labels)
        train_losses.append(loss.detach().item())
        y_predict_prob.extend(output.cpu().detach().numpy())
        y_true.extend(cur_labels.cpu().numpy())

        if epoch % 5 == 0:
            print("Epoch {} - batch {}: loss = {}".format(epoch, i, loss))

        # Backpropogation and update weights
        loss.backward()
        optimizer.step()

    # Convert tensor to numpy array so we can use sklearn's metrics
    y_predict_prob = np.stack(y_predict_prob)
    y_predict = np.argmax(y_predict_prob, axis=1)
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
            cur_batch, cur_labels = cur_batch.to(device), cur_labels.to(device)
            output = model(cur_batch)

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

    # Set model to evaluation mode
    model.eval()

    test_losses, y_predict_prob, y_true = [], [], []

    with torch.no_grad():
        for cur_batch, cur_labels in test_generator:
            # Even there is only forward() in testing phase, it is still faster
            # to do it on GPU
            cur_batch, cur_labels = cur_batch.to(device), cur_labels.to(device)

            output = model(cur_batch)
            loss = criterion(output, cur_labels)

            # Track the loss and prediction for each batch
            test_losses.append(loss.detach().item())
            y_predict_prob.extend(output.cpu().detach().numpy())
            y_true.extend(cur_labels.cpu().numpy())

    # Convert tensor to numpy array so we can use sklearn's metrics
    # sklearn loves 1d proba array of the activated class
    y_predict_prob = np.stack(y_predict_prob)[:, 1]
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


def generate_data(bs, nproc, img_dir='./images'):
    params = {
        'batch_size': bs,
        'shuffle': True,
        'num_workers': nproc
    }

    img_names = glob(join(img_dir, '*.npz'))

    # Randomly split img_names into three sets by 6:2:2
    img_names = shuffle(img_names)
    quintile_len = len(img_names) // 5
    vali_names = img_names[: quintile_len]
    test_names = img_names[quintile_len: quintile_len * 2]
    train_names = img_names[quintile_len * 2:]

    print(("There are {} training samples, {} validation samples, " +
           "and {} test samples.").format(
               len(train_names), len(vali_names), len(test_names)))

    # Create data generators
    training_dataset = Dataset(train_names)
    training_generator = data.DataLoader(training_dataset, **params)

    vali_dataset = Dataset(vali_names)
    vali_generator = data.DataLoader(vali_dataset, **params)

    test_dataset = Dataset(test_names)
    test_generator = data.DataLoader(test_dataset, **params)

    return training_generator, vali_generator, test_generator


def train_main(assay, lr, bs, nproc, patience, epoch, img_dir='./images'):

    # Generate three datasets
    training_generator, vali_generator, test_generator = generate_data(
        bs, nproc, img_dir=img_dir
    )

    # Run on GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on {}.".format(device))

    # Use cross-entropy as our loss funciton
    lenet = LeNet()
    lenet.to(device)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(lenet.parameters(), lr=0.001)

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
            lenet,
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
            break

    # After training, evaluate trained model on the test set
    test_loss, test_acc, y_predict_prob, y_true = test(
        lenet, device, criterion, test_generator
    )

    # Save weights, results
    torch.save(lenet.state_dict(), './trained_weights_{}_{}_{}.pt'.format(
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
    epoch = 2
    patience = 20
    img_dir = './images'
    train_main(assay, lr, bs, nproc, patience, epoch, img_dir=img_dir)
