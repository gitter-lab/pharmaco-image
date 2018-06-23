import keras
import sys
import numpy as np
from keras.models import Model
from keras import optimizers
from keras import layers
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.callbacks import EarlyStopping
from keras.layers import Activation, BatchNormalization, Conv2D, \
    AveragePooling2D
from sklearn.model_selection import train_test_split
from os.path import join, basename, exists
from glob import glob
from json import dump


class DataGenerator(keras.utils.Sequence):
    """
    Implement our data generator for the fit_generator() call.
    Its document says:
    > Every Sequence must implement the __getitem__ and the __len__ methods.
    > If you want to modify your dataset between epochs you may implement
    > on_epoch_end. The method __getitem__ should return a complete batch.
    """
    def __init__(self, bottlenecks, labels, channel, batch_size=32,
                 dim=(299, 299, 3), num_classes=2):
        """
        args:
            bottlenecks: array of the bottleneck path
            labels: array of corresponding labels
            channel: 0 -> channel 123, 1 -> channel 45
            batch_size: number of training samples per epoch
            dim: the default is for inception v3
            num_classes: the default is for binary classification
        """

        self.bottlenecks = bottlenecks
        self.labels = labels
        self.channel = channel
        self.batch_size = batch_size
        self.dim = dim
        self.num_classes = num_classes
        self.indexes = np.arange(len(self.bottlenecks))

    def __len__(self):
        """
        This method tells keras how many times to go through the whole sample.
        """
        return int(np.ceil(len(self.bottlenecks) / float(self.batch_size)))

    def __getitem__(self, index):
        """
        This method generates one batch of data.
        """
        batch_indexes = self.indexes[index * self.batch_size:
                                     (index + 1) * self.batch_size]

        # Get the corresponding bottleneck path
        batch_names = [self.bottlenecks[i] for i in batch_indexes]
        batch_x = np.vstack([get_bottleneck_feature(p, self.channel)
                             for p in batch_names])

        # Load the labels
        batch_label = np.array([self.labels[i] for i in batch_indexes])
        batch_y = keras.utils.to_categorical(batch_label,
                                             num_classes=self.num_classes)

        return batch_x, batch_y


def partition_data(data_dir, channel, batch_size=32, train_percentage=0.6,
                   vali_percentage=0.2, test_percentage=0.2):
    """
    Partition the data into train, validation and test set based on the given
    proportion. Each proportion should be a float in (0, 1).

    Return three keras data generator (DataGenerator).
    """
    # We first load all the image names and labels
    names = []
    labels = []
    cur_label = 0

    # Also store a label map in case in the future we will classify multiple
    # classes
    label_mapping = {}
    split_mapping = {}

    for sub_dir in glob(join(data_dir, '*')):
        # Update the mapping
        label_mapping[basename(sub_dir)] = cur_label

        # Load all the bottlenecks
        for bottle_name in glob(join(sub_dir, "*.npz")):
            names.append(bottle_name)
            labels.append(cur_label)

        # Update to the next label
        cur_label += 1

    # Partition to train, valid and test sets
    name_train, name_temp, label_train, label_temp = train_test_split(
        np.array(names),
        np.array(labels, dtype=int),
        test_size=vali_percentage + test_percentage,
        random_state=42
    )

    name_vali, name_test, label_vali, label_test = train_test_split(
        name_temp,
        label_temp,
        test_size=test_percentage / (test_percentage + vali_percentage),
        random_state=42
    )

    split_mapping['train'] = name_train.tolist()
    split_mapping['vali'] = name_vali.tolist()
    split_mapping['test'] = name_test.tolist()

    # Encode each partition into the DataGenerator
    nc = len(label_mapping)
    return DataGenerator(name_train, label_train, channel,
                         batch_size=batch_size, num_classes=nc), \
        DataGenerator(name_vali, label_vali, channel,
                      batch_size=batch_size, num_classes=nc), \
        DataGenerator(name_test, label_test, channel,
                      batch_size=batch_size, num_classes=nc), \
        label_mapping, \
        split_mapping


def retrain(num_class, train_generator, vali_generator, epoch=1,
            nproc=1, lr=0.001, save_model_path=None, class_weights=None,
            train_model=None, early_stopping_patience=None):
    """
    Retrain the inception v3 model's top layer.

    Retraining on bottleneck features if and only if train_model is given.
    """
    if not train_model:
        # Change inception input layer to use gray scale images
        input_tensor = Input(shape=(299, 299, 3))
        # Not including the top classification layer
        base_model = InceptionV3(input_tensor=input_tensor,
                                 weights='imagenet',
                                 include_top=False)

        # Add the new training layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        # Logit layer
        predictions = Dense(num_class, activation='softmax')(x)

        # Add those new layers into our model
        train_model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze all Inception_V3 layers
        for layer in base_model.layers:
            layer.trainable = False

    # Compile the model
    train_model.compile(optimizer=optimizers.Adam(lr=lr,
                                                  beta_1=0.9,
                                                  beta_2=0.999,
                                                  epsilon=None,
                                                  decay=0.0),
                        metrics=['accuracy'],
                        loss='categorical_crossentropy')

    if not class_weights:
        # If class weights are not given, then all have the same weight
        class_weights = {}
        for i in range(num_class):
            class_weights[i] = 1.0

    # Add early stopping
    early_stopping = None
    if early_stopping_patience:
        early_stopping = [EarlyStopping(monitor='val_loss',
                                        patience=early_stopping_patience)]

    # train the model on the new data
    # ❗️Caveat: the old version of Keras requires argument steps_per_epoch
    # and validation_steps
    hist = train_model.fit_generator(generator=train_generator,
                                     validation_data=vali_generator,
                                     steps_per_epoch=len(train_generator),
                                     validation_steps=len(vali_generator),
                                     epochs=epoch,
                                     callbacks=early_stopping,
                                     class_weight=class_weights,
                                     verbose=1,
                                     use_multiprocessing=True,
                                     workers=nproc)

    if save_model_path:
        train_model.save(save_model_path)

    return hist


def get_bottleneck_feature(bottle_path, channel):
    """
    Load the bottleneck feature of the image from the cached file.
    Args:
        bottle_path: path to the npz file
        channel: 0 -> chanel 123, 1 -> chanel 45
    """
    # Check if the bottleneck has been created.
    if(not exists(bottle_path)):
        print("Failed to load bottleneck, {} doesn't exist".format(
            bottle_path))
        return

    # Load the bottleneck feature
    bottleneck_features = np.load(bottle_path)['feature']

    # Each bottleneck file has two features
    if channel == 0:
        bottleneck_features = bottleneck_features[:, :2048]
    else:
        bottleneck_features = bottleneck_features[:, 2048:]

    return bottleneck_features


# This function is copied from Keras's implementation of Inception_v3
# https://github.com/keras-team/keras/blob/
# b0f1bb9c7c68e24137a9dc284dc210eb0050a6b4/keras/applications/inception_v3.py
def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def create_model_one_layer(num_classes):
    """
    Create the model we were using in TF version.
    """
    # Create bottleneck model
    base_model = InceptionV3(weights='imagenet', include_top=True)
    bottleneck_model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("avg_pool").output
    )

    # Create training model
    bottleneck_input = Input(shape=(2048, ))
    predictions = Dense(num_classes, activation='softmax')(bottleneck_input)
    train_model = Model(inputs=bottleneck_input,
                        outputs=predictions)

    return bottleneck_model, train_model


def create_model_two_layers(num_classes):
    """
    Retrain the last two mixed_9 and mixed_10 layers.

    I thought there is a simple way to copy existing layer in Keras, but I am
    wrong. The best idea now is to re-construct those layers we want to retrain
    into our train model.

    Another way is to use the original model by setting other layers to
    untrainable. However, it is not taking advantage of bottleneck cache (way
    too slow).
    """

    channel_axis = 3
    # Create bottleneck model
    base_model = InceptionV3(weights='imagenet', include_top=True)
    bottleneck_model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("activation_86").output
    )

    # Create training model
    bottleneck_input = Input(shape=(8, 8, 320))

    # The construct of mixed_9 and mixed_10 is copied from Keras's
    # implementation of Inception_v3
    # https://github.com/keras-team/keras/blob/
    # b0f1bb9c7c68e24137a9dc284dc210eb0050a6b4/keras/
    # applications/inception_v3.py

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(bottleneck_input, 320, 1, 1)

        branch3x3 = conv2d_bn(bottleneck_input, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis, name='mixed9_' + str(i)
        )

        branch3x3dbl = conv2d_bn(bottleneck_input, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis
        )

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same'
        )(bottleneck_input)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i)
        )

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    train_model = Model(inputs=bottleneck_input,
                        outputs=predictions)

    return bottleneck_model, train_model


if __name__ == "__main__":
    # Load constants
    train_percentage = 0.8
    vali_percentage = 0.2
    test_percentage = 0

    data_dir = "./features_24277"
    channel = 0

    epoch = 100
    patience = 100
    batch_size = 8
    save_model_path = './train_output/model_{}.h5'.format(channel)
    nproc = 8
    lr = 0.5

    if len(sys.argv) > 1:
        channel = sys.argv[1]

    # Partition and load our images
    train, vali, test, mapping, split_mapping = partition_data(
        data_dir,
        channel,
        train_percentage=train_percentage,
        vali_percentage=vali_percentage,
        test_percentage=test_percentage,
        batch_size=batch_size,
    )

    # The dataset is highly unbalanced, so we want to weight down the weight
    # of dmso
    num_classes = len(mapping)
    class_weights = {}
    for i in range(num_classes):
        class_weights[i] = 1.0
    class_weights[mapping['24277_dmso']] = 0.15

    bottleneck_model, train_model = create_model_one_layer(num_classes)

    # Start training
    hist = retrain(num_classes, train, vali,
                   epoch=epoch, nproc=nproc, lr=lr,
                   save_model_path=save_model_path, train_model=train_model,
                   class_weights=class_weights,
                   early_stopping_patience=patience)

    # Save the training statistics for analysis
    dump(hist.history, open("./train_output/history.json", 'w'), indent=4)

    # Save the label mapping
    dump(mapping, open("./train_output/labels.json", 'w'), indent=4)

    # Save the partition mapping
    dump(split_mapping, open("./train_output/split.json", 'w'), indent=4)
