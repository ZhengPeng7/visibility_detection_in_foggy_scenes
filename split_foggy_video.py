import cv2
import numpy as np
import os
from keras.models import model_from_json
from utils import extract_sec_region
import time
import matplotlib.pyplot as plt


time_st = time.time()
# settings
FRAMES_PER_SECOND = 24
FRAMES_SAVE_PER_MINUTE = 60
VALID_SECOND_DIGIT = [0, 5]
dst_root_dir = 'split_frames_new'
# load digit recognition model
with open('./digit_recognizer/structure.json', 'r') as fout:
    structure = fout.read()
digit_recognizer = model_from_json(structure)
digit_recognizer.load_weights('./digit_recognizer/weights_new.hdf5')

video_dir = 'fog_videos'
videos = sorted(
    os.listdir(video_dir),
    key=lambda x: int(x.split('_')[0])
)
video_paths = [os.path.join(video_dir, i) for i in videos]

sec_regions = [
    [35, 71, 646, 671],
    [35, 83, 861, 895], [35, 83, 861, 895], [35, 83, 861, 895], [35, 83, 861, 895], [35, 83, 861, 895]
]
file_num = [200, 230, 230, 180, 150, 130]

print('videos:', videos)
for v in range(len(videos))[:]:
    cap = cv2.VideoCapture(video_paths[v])
    file_idx = 0
    frame_idx = 0
    if not os.path.exists(os.path.join(dst_root_dir, videos[v].split('.')[0])):
        os.mkdir(os.path.join(dst_root_dir, videos[v].split('.')[0]))
    
    # Get the start point
    for _ in range(FRAMES_PER_SECOND * 10):
        ret, frame = cap.read()
        frame_idx += 1
        sec_region = extract_sec_region(frame, sec_regions[v])
        sec_num = np.argmax(digit_recognizer.predict(np.asarray([sec_region])), axis=1).ravel()[0]
        print('sn:', sec_num)
        if sec_num in VALID_SECOND_DIGIT:
            for _ in range(int(FRAMES_PER_SECOND * 0.5)):
                ret, frame = cap.read()
                frame_idx += 1
            break
    idx_suffix = str(file_idx // FRAMES_SAVE_PER_MINUTE) + '_' + str(file_idx % FRAMES_SAVE_PER_MINUTE)
    print('The first valid frame.')
    sec_num = np.argmax(digit_recognizer.predict(np.asarray([sec_region])), axis=1).ravel()[0]
    print('sn:', sec_num)
    print(os.path.join(dst_root_dir, videos[v].split('.')[0], ''.join((videos[v].split('.')[0], '_', idx_suffix, '.jpg'))))
    cv2.imwrite(os.path.join(dst_root_dir, videos[v].split('.')[0], ''.join((videos[v].split('.')[0], '_', idx_suffix, '.jpg'))), frame)
    file_idx += 1

    while cap.isOpened():
        # --> 4s
        for idx_skip in range(1, FRAMES_PER_SECOND * 4 + 1):
            ret, frame = cap.read()
            if idx_skip % FRAMES_PER_SECOND == 0:
                idx_suffix = str(file_idx // FRAMES_SAVE_PER_MINUTE) + '_' + str(file_idx % FRAMES_SAVE_PER_MINUTE)
                cv2.imwrite(os.path.join(dst_root_dir, videos[v].split('.')[0], ''.join((videos[v].split('.')[0], '_', idx_suffix, '.jpg'))), frame)
                file_idx += 1
            frame_idx += 1
        sec_num = np.argmax(digit_recognizer.predict(np.asarray([extract_sec_region(frame, sec_regions[v])])))
        print('sec_num:', sec_num)
        # if not 0 or 5, --> more seconds
        if FRAMES_PER_SECOND * (5 - sec_num % 5) > 0:
            print('{} more frames.'.format(FRAMES_PER_SECOND * (5 - sec_num % 5)))
        sec_num_bk = sec_num
        while sec_num not in VALID_SECOND_DIGIT:
            for _ in range(FRAMES_PER_SECOND * (5 - sec_num % 5)):
                ret, frame = cap.read()
                frame_idx += 1
                if _ % 12 == 0:
                    sec_num = np.argmax(digit_recognizer.predict(np.asarray([extract_sec_region(frame, sec_regions[v])])))
                    if sec_num in VALID_SECOND_DIGIT or sec_num % 5 < sec_num_bk % 5:
                        break

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or file_idx >= file_num[v] * FRAMES_SAVE_PER_MINUTE:
            break

        idx_suffix = str(file_idx // FRAMES_SAVE_PER_MINUTE) + '_' + str(file_idx % FRAMES_SAVE_PER_MINUTE)
        print(os.path.join(dst_root_dir, videos[v].split('.')[0], ''.join((videos[v].split('.')[0], '_', idx_suffix, '.jpg'))))
        cv2.imwrite(os.path.join(dst_root_dir, videos[v].split('.')[0], ''.join((videos[v].split('.')[0], '_', idx_suffix, '.jpg'))), frame)
        file_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()

print('Duration:', time.time() - time_st)
