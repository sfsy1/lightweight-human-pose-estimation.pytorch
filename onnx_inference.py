import matplotlib.pyplot as plt
import cv2
import numpy as np

from scripts.simplified_functions import sigmoid, normalize, find_distance, find_peaks, choose_best_peak, \
                                        get_keypoint_locations, get_signals_from_keypoints, \
                                        get_scores_from_signals, preprocess_image

import onnx
import onnxruntime as ort

# We will only be using 6 of the key points
kpt_names = ['nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 
             'l_wri', 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee',
             'l_ank', 'r_eye', 'l_eye', 'r_ear', 'l_ear']

# points of interest 
poi = [0, 1, 2, 5, 14, 15]
print([kpt_names[i] for i in poi])

# default positions (y,x) of different parts with the origin on the top left
default_keypoints = [(0.4,0.5), (0.75,0.5), (0.75,0.3), (0.75,0.7),(0.35,0.45), (0.35,0.55)]

# they're really just telling the algorithm roughly where to look for these keypoints by default
plt.figure(figsize=(3,2)); plt.scatter([x[1] for x in default_keypoints], [-x[0] for x in default_keypoints]);

# We will only be using 6 of the key points
kpt_names = ['nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 
             'l_wri', 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee',
             'l_ank', 'r_eye', 'l_eye', 'r_ear', 'l_ear']


# points of interest 
poi = [0, 1, 2, 5, 14, 15]

# default positions (y,x) of different parts with the origin on the top left
default_keypoints = [(0.4,0.5), (0.75,0.5), (0.75,0.3), (0.75,0.7),(0.35,0.45), (0.35,0.55)]

calib_keypoints = default_keypoints



sess = ort.InferenceSession('pose360smallfloat16.onnx')

# img = cv2.imread('calibration/test2.jpg')
# img = cv2.imread('../tw_onnxruntime/csharp/sample/Microsoft.ML.OnnxRuntime.ResNet50v2Sample/bin/Debug/netcoreapp3.1/test11.jpg')
img = cv2.imread('image.jpg')

input_tensor = preprocess_image(img)

output_tensor = sess.run(None, {'input': input_tensor})

keypoint_locations = get_keypoint_locations(output_tensor, poi, calib_keypoints)

signals = get_signals_from_keypoints(keypoint_locations)

"""
5. scoring by comparing signals with calibration signals (4 scores from 0-1)
"""
calib_signals = get_signals_from_keypoints(calib_keypoints)
scores = get_scores_from_signals(signals, calib_signals)

"""
6. average the 4 scores to get final score (from 0-1)
"""
final_score = sum(scores)/4
print(str(final_score))

import time
time.sleep(3)

file = open("final_score.txt", "w") 
file.write(str(final_score)) 
file.close() 