import cv2
import os
import numpy as np


def insert_missing_frame(dir_path, frames_dir, missing_frame_paths, insert_indexes):
    idx_imread = 0
    files = sorted(os.listdir(frames_dir), key=lambda x: int(x[:-4].split('_')[-2]) * 12 + int(x[:-4].split('_')[-1]))
    for idx_missing_frame in insert_indexes:
        latter_half = [os.path.join(frames_dir, f) for f in files[int(idx_missing_frame.split('_')[0]) * 12 + int(idx_missing_frame.split('_')[1]) + idx_imread:-1]]
        # os.remove(latter_half[-1])
        for idx_mf in range(len(latter_half)-1, 0, -1):
            os.rename(latter_half[idx_mf-1], latter_half[idx_mf])
        cv2.imwrite(latter_half[0], cv2.imread(missing_frame_paths[idx_imread]))
        idx_imread += 1

    return None


def main():
    insert_index_lst = [['95_3', '185_6'], ['17_8', '65_7'], ['159_5'], [], [], ['88_8']]
    missing_frame_paths = [
        ['/home/chengxg/Pictures/1_point_05_95_3.jpg', '/home/chengxg/Pictures/1_point_05_185_6.jpg'],
        ['/home/chengxg/Pictures/2_point_17E_17_8.jpg', '/home/chengxg/Pictures/2_point_17E_65_7.jpg'],
        ['/home/chengxg/Pictures/3_point_17G_0414_159_5.jpg'],
        [],
        [],
        ['/home/chengxg/Pictures/6_point_16B_0315_88_8.jpg']
    ]
    dir_path = 'split_frames'
    frames_dirs = [os.path.join(dir_path, p) for p in sorted(os.listdir(dir_path))]
    print('frames_dirs:', len(frames_dirs), frames_dirs)
    for idx_dir in range(len(frames_dirs)):
        print('Inserting {}...'.format(missing_frame_paths[idx_dir]))
        insert_missing_frame(dir_path, frames_dirs[idx_dir], missing_frame_paths[idx_dir], insert_index_lst[idx_dir])

if __name__ == '__main__':
    main()