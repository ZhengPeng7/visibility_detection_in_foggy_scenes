import cv2
import numpy as np
from math import floor, ceil
import os
import keras
from keras.applications.vgg16 import preprocess_input
import keras.backend as K


def generate_generator(img_paths, labels, batch_size=32, net='vgg', reverse=False):
    ## Replace the original square input_size(like 224x224, 299x299) with a new input_size, in order to try to make the input_size similar to the input image's shape.
    input_size = (1920//6, (890 - 95)//6)
    if 'resnet' in net:
        ## Since resnet has a requirement that the input shape must be no smaller than (197, 197), I set its input_size specially as (320, 200).
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

            if img_paths[idx_total].split('/')[-1].split('_')[0] == str(1):
                ## Cut the top and bottom horizontally,
                ## the size of images in 1st scene(1280x720) is smaller than others(1920x1080).
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
                ## Flip the image horizontally to double the amount of data.
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
            # x = preprocess_input(x, mode='tf')
            yield x, y


def gen_img_paths_and_labels(img_dir_paths, label_paths, validate_scene_idx=3, normalize_label=False):
    
    img_paths = []
    validate_paths = []
    for scene in os.listdir(img_dir_paths):
        files_sorted = sorted(os.listdir(os.path.join(img_dir_paths, scene)), key=lambda x: int(x.strip('.jpg').split('_')[-2]) * 12 + int(x.strip('.jpg').split('_')[-1]))
        # print('files_sorted:\n', files_sorted[:20])
        if int(scene.split('_')[0]) == validate_scene_idx:
            for f in files_sorted:
                validate_paths += [os.path.join(img_dir_paths, scene, f)]
        else:
            for f in files_sorted:
                img_paths += [os.path.join(img_dir_paths, scene, f)]
            # for _ in range(2):
            #     for f in files_sorted[1500:]:
            #         img_paths += [os.path.join(img_dir_paths, scene, f)]
    # vis = scio.loadmat('./data/Subject_Measured_Vis.mat')
    labels = []
    validate_labels = []
    labels_all = []
    for k in os.listdir(label_paths):# vis.keys():
    #     if 'Subject_Vis_mean' in k:
        label_curr = np.loadtxt(os.path.join(label_paths, k)).tolist()
        if str(validate_scene_idx) in k:
    #         if int(k.strip('Subject_Vis_mean').split('_')[0]) == validate_scene_idx:
            validate_labels += label_curr
        else:
            labels += label_curr
            # for _ in range(2):
            #     labels += label_curr[1500:]
        labels_all += np.loadtxt(os.path.join(label_paths, k)).tolist()
    (label_min, label_max) = (min(labels_all), max(labels_all)) if normalize_label else (0, 1)
    labels = (np.vstack(labels).ravel() - label_min) / (label_max - label_min)
    validate_labels = (np.vstack(validate_labels).ravel() - label_min) / (label_max - label_min)
    return img_paths, validate_paths, labels, validate_labels, label_min, label_max


def loss_APE(labels, preds):
    return K.mean(K.abs(preds - labels) / labels ,axis=1) * 100.


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_train_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))


class SaveModelOnMAPE(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.APE = round(logs.get('val_loss'), 3)
        if self.APE < 30 or 1:
            self.model.save_weights(os.path.join('weights_new', self.model.name, self.model.name + '_MAPE_epoch' + str(epoch) + '_MAPE' + str(self.APE) + '.hdf5'))


class SaveModelOnMSE(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.MSE = round(logs.get('val_loss'), 3)
        if self.MSE < 2000 or 1:
            self.model.save_weights(os.path.join('weights_new', self.model.name, self.model.name + '_MSE_epoch' + str(epoch) + '_MSE' + str(self.MSE) + '.hdf5'))


class SaveModelOnMAE(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.MAE = round(logs.get('val_loss'), 3)
        if self.MAE < 40 or 1:
            self.model.save_weights(os.path.join('weights_new', self.model.name, self.model.name + '_MAE_epoch' + str(epoch) + '_MAE' + str(self.MAE) + '.hdf5'))



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
                        ),
                        127,
                        255,
                        cv2.THRESH_OTSU
                    )[1],
                    ((0, 0), (floor(supply_len), ceil(supply_len))),
                    'constant',
                    constant_values=((0, 0), (255, 255))
                ),
                (36, 36),
                interpolation=cv2.INTER_LANCZOS4
            ),
            127,
            255,
            cv2.THRESH_OTSU
        )[1],
        cv2.COLOR_GRAY2BGR
    )
    return sec_digit_region
