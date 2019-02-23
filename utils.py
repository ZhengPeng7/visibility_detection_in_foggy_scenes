import cv2
import numpy as np
from math import floor, ceil
import os
import keras
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
import pandas as pd
from sklearn.utils import shuffle


def get_data(idx_validate):
    split_idx = (np.asarray([0, 2400, 5160, 7920, 10080, 11880, 13440])*5).tolist()
    data = pd.read_excel('./imgPaths_labels_isSensor_new.xlsx')
    paths, labels, sensor = data['img_path'], data['label'], data['sensor']
    paths = np.squeeze(np.asarray(paths)).tolist()
    labels = np.squeeze(np.asarray(labels)).tolist()
    sensor = np.squeeze(np.asarray(sensor)).tolist()
    train_paths = paths[:split_idx[idx_validate-1]] + paths[split_idx[idx_validate]:]
    test_paths = paths[split_idx[idx_validate-1]:split_idx[idx_validate]]
    train_labels = labels[:split_idx[idx_validate-1]] + labels[split_idx[idx_validate]:]
    test_labels = labels[split_idx[idx_validate-1]:split_idx[idx_validate]]
    train_sensor = sensor[:split_idx[idx_validate-1]] + sensor[split_idx[idx_validate]:]
    test_sensor = sensor[split_idx[idx_validate-1]:split_idx[idx_validate]]

    test_paths = np.asarray(test_paths).reshape(len(test_paths), 1)
    test_labels= np.asarray(test_labels).reshape(len(test_labels), 1)

    idx_real = np.where(np.asarray(test_sensor) == 1)
    idx_test = np.loadtxt('./indices/indices_test_or_5_val_{}.txt'.format(idx_validate)).astype(int)
    test_paths_real = test_paths[idx_real]
    test_labels_real = test_labels[idx_real]
    test_paths_for_test = np.squeeze(test_paths[idx_test]).tolist()
    test_labels_for_test = np.squeeze(test_labels[idx_test]).tolist()
    idx_val = np.loadtxt('./indices/indices_rand_val_no_5_val_{}.txt'.format(idx_validate)).astype(int)
    validate_paths = test_paths[idx_val]
    validate_labels = test_labels[idx_val]

    test_paths = np.squeeze(test_paths).tolist()
    test_labels = np.squeeze(test_labels).tolist()
    test_paths_real = np.squeeze(test_paths_real).tolist()
    test_labels_real = np.squeeze(test_labels_real).tolist()

    validate_paths, validate_labels = np.squeeze(validate_paths).tolist(), np.squeeze(validate_labels).tolist()

    idx_train_rand = shuffle(range(len(train_labels)))
    train_paths = np.asarray(train_paths)[idx_train_rand].tolist()
    train_labels = np.asarray(train_labels)[idx_train_rand].tolist()
    return train_paths, train_labels, validate_paths, validate_labels, test_paths, test_labels, test_paths_real, test_labels_real, test_paths_for_test, test_labels_for_test


def generate_generator(img_paths, labels, batch_size=32, net='inceptionV4', reverse=False):
    # Resize original images into their (1/6, 1/6) sizes,
    # in order to try to keep the shape of input images.
    input_size = (1920//6, (890 - 95)//6)
    if 'resnet' in net:
        ## Since resnet has a requirement that the input shape must be no smaller than (197, 197),
        # I set its input_size specially as (320, 200).
        input_size = (320, 200)
    data_len = len(labels) if isinstance(labels, list) else labels.shape[0]
    flag_continue = 0
    idx_total = 0
    while True:
        if not flag_continue:
            x = []
            y = []
            inner_iter_num = batch_size
        else:
            idx_total = 0
            inner_iter_num = batch_size - data_len % batch_size
        for _ in range(inner_iter_num):
            if idx_total >= data_len:
                flag_continue = 1
                break
            else:
                flag_continue = 0

            # Cut the top and bottom horizontally, since these parts can't
            # make contribution to the training, while some disturbation.
            if img_paths[idx_total].split('/')[-1].split('_')[0] == str(1):
                # The shape of images in Scene 1(1280x720) is different from others(1920x1080).
                img = cv2.imread(img_paths[idx_total])[80:525, :, :]
            else:
                img = cv2.imread(img_paths[idx_total])[95:890]
            x.append(
                cv2.cvtColor(
                    cv2.resize(img, input_size, interpolation=cv2.INTER_LANCZOS4),
                    cv2.COLOR_BGR2RGB
                )
            )
            y.append(labels[idx_total])
            if reverse:
                # Flip the image horizontally to double the amount of data.
                x.append(
                    cv2.cvtColor(
                        cv2.resize(cv2.flip(img, 1), input_size, interpolation=cv2.INTER_LANCZOS4),
                    cv2.COLOR_BGR2RGB
                    )
                )
                y.append(labels[idx_total])
            idx_total += 1
        if not flag_continue:
            x, y = np.asarray(x).astype(np.float), np.asarray(y)
            yield x, y


def gen_img_paths_and_labels(img_dir_paths, label_paths, validate_scene_idx=3, normalize_label=False):
    img_paths = []
    validate_paths = []
    for scene in os.listdir(img_dir_paths):
        files_sorted = sorted(
            os.listdir(os.path.join(img_dir_paths, scene)),
            key=lambda x: int(x.strip('.jpg').split('_')[-2]) * 12 + int(x.strip('.jpg').split('_')[-1])
        )
        # print('files_sorted:\n', files_sorted[:20])
        if int(scene.split('_')[0]) == validate_scene_idx:
            for f in files_sorted:
                validate_paths += [os.path.join(img_dir_paths, scene, f)]
        else:
            for f in files_sorted:
                img_paths += [os.path.join(img_dir_paths, scene, f)]
            # Train on certain parts of data-image for 2 more times.
            # for _ in range(2):
            #     for f in files_sorted[1500:]:
            #         img_paths += [os.path.join(img_dir_paths, scene, f)]
    # vis = scio.loadmat('./data/Subject_Measured_Vis.mat')
    labels = []
    validate_labels = []
    labels_all = []
    for k in os.listdir(label_paths):
        label_curr = np.loadtxt(os.path.join(label_paths, k)).tolist()
        if str(validate_scene_idx) in k:
            validate_labels += label_curr
        else:
            labels += label_curr
            # Train on certain parts of data-label for 2 more times.
            # for _ in range(2):
            #     labels += label_curr[1500:]
        labels_all += np.loadtxt(os.path.join(label_paths, k)).tolist()
    (label_min, label_max) = (min(labels_all), max(labels_all)) if normalize_label else (0, 1)
    labels = (np.vstack(labels).ravel() - label_min) / (label_max - label_min)
    validate_labels = (np.vstack(validate_labels).ravel() - label_min) / (label_max - label_min)
    return img_paths, validate_paths, labels, validate_labels, label_min, label_max


def loss_MAPE(labels, preds):
    return K.mean(K.abs(preds - labels) / labels ,axis=1) * 100.


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_train_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))


class SaveModelOnMAPE_1(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.MAPE = round(logs.get('val_loss'), 3)
        if self.MAPE < 30 or 1:
            self.model.save_weights(os.path.join('weights_new', 'MAPE', self.model.name, self.model.name + '_MAPE_epoch' + str(epoch) + '_MAPE' + str(self.MAPE) + '.hdf5'))


class SaveModelOnMAPE_2(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.MAPE = round(logs.get('val_loss'), 3)
        if self.MAPE < 30 or 1:
            self.model.save_weights(os.path.join('weights_new', 'MAPE', self.model.name, self.model.name + '_MAPE_epoch' + str(epoch+60) + '_MAPE' + str(self.MAPE) + '.hdf5'))


class SaveModelOnMSE_1(keras.callbacks.Callback):
    def __init__(self, idx_test_set):
        self.idx_test_set = idx_test_set
    def on_epoch_end(self, epoch, logs={}):
        self.MSE = round(logs.get('val_loss'), 3)
        if self.MSE < 2000 or 1:
            self.model.save_weights(os.path.join('weights_new', 'MSE', self.model.name, 'Test_set_{}'.format(self.idx_test_set), self.model.name + '_MSE_epoch' + str(epoch) + '_MSE' + str(self.MSE) + '.hdf5'))


class SaveModelOnMSE_2(keras.callbacks.Callback):
    def __init__(self, idx_test_set):
        self.idx_test_set = idx_test_set
    def on_epoch_end(self, epoch, logs={}):
        self.MSE = round(logs.get('val_loss'), 3)
        if self.MSE < 2000 or 1:
            self.model.save_weights(os.path.join('weights_new', 'MSE', self.model.name, 'Test_set_{}'.format(self.idx_test_set), self.model.name + '_MSE_epoch' + str(epoch+60) + '_MSE' + str(self.MSE) + '.hdf5'))


class SaveModelOnMAE_1(keras.callbacks.Callback):
    def __init__(self, idx_test_set):
        self.idx_test_set = idx_test_set
    def on_epoch_end(self, epoch, logs={}):
        self.MAE = round(logs.get('val_loss'), 3)
        if self.MAE < 40 or 1:
            self.model.save_weights(os.path.join('weights_new', 'MAE', self.model.name, 'Test_set_{}'.format(self.idx_test_set), self.model.name + '_MAE_epoch' + str(epoch) + '_MAE' + str(self.MAE) + '.hdf5'))


class SaveModelOnMAE_2(keras.callbacks.Callback):
    def __init__(self, idx_test_set):
        self.idx_test_set = idx_test_set
    def on_epoch_end(self, epoch, logs={}):
        self.MAE = round(logs.get('val_loss'), 3)
        if self.MAE < 40 or 1:
            self.model.save_weights(os.path.join('weights_new', 'MAE', self.model.name, 'Test_set_{}'.format(self.idx_test_set), self.model.name + '_MAE_epoch' + str(epoch+60) + '_MAE' + str(self.MAE) + '.hdf5'))



# Split frames module
def extract_sec_region(frame, sec_region):
    supply_len = np.sum(np.asarray(sec_region)*[-1, 1, 1, -1])/2
    sec_digit_region = cv2.cvtColor(
        cv2.threshold(
            cv2.resize(
                np.pad(
                    cv2.threshold(
                        cv2.cvtColor(
                            frame[sec_region[0]:sec_region[1], sec_region[2]:sec_region[3]],
                            cv2.COLOR_BGR2GRAY
                        ), 127, 255, cv2.THRESH_OTSU
                    )[1],
                    ((0, 0), (floor(supply_len), ceil(supply_len))),
                    'constant', constant_values=((0, 0), (255, 255))
                ),
                (36, 36),
                interpolation=cv2.INTER_LANCZOS4
            ), 127, 255, cv2.THRESH_OTSU
        )[1],
        cv2.COLOR_GRAY2BGR
    )
    return sec_digit_region
