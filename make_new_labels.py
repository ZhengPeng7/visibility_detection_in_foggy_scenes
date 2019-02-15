import pandas as pd
import os
from utils import gen_img_paths_and_labels
import numpy as np


FRAMES_PER_MINUTE = 60
idx_start = [2*5, 19*5, 9*5, 9*5, 13*5, 11*5]
paths = []
labels = []
is_sensor_all = []
for i in range(1, 7, 1):
    img_dir_paths, label_paths, validate_scene_idx = './split_frames_new/', 'labels_new', i
    img_paths, validate_paths, train_labels, validate_labels, label_min, label_max = gen_img_paths_and_labels(img_dir_paths, label_paths, validate_scene_idx)
    paths += validate_paths
    labels += validate_labels.tolist()
    print(i, len(labels), len(paths))
    is_sensor = np.zeros((len(validate_paths))).tolist()
    for j in range(len(is_sensor)):
        if j % FRAMES_PER_MINUTE == (idx_start[i-1] % FRAMES_PER_MINUTE) and j > idx_start[i-1]:
            is_sensor[j] = 1
    is_sensor_all += is_sensor

paths, labels, is_sensor_all = np.asarray(paths).reshape(-1, 1), np.asarray(labels).reshape(-1, 1), np.asarray(is_sensor_all).reshape(-1, 1).astype(np.int)
index = np.asarray(list(range(paths.shape[0]))).reshape(-1, 1)

print('index.shape:', index.shape)
print('paths.shape:', paths.shape)
print('labels.shape:', labels.shape)
print('is_sensor_all:', is_sensor_all.shape)

res = np.hstack([index, paths, labels, is_sensor_all])

xlsx_writer = pd.ExcelWriter('./imgPaths_labels_isSensor_new.xlsx')
pd.DataFrame(res).to_excel(xlsx_writer, 'Sheet1', header=['index', 'img_path', 'label', 'sensor'], index=None)
xlsx_writer.save()

