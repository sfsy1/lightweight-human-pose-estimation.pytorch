import numpy as np
import cv2
    
    
def sigmoid(a,b,c,d,x):
    return a/( 1 + np.exp(b - x/c) ) + d


def find_distance(a,b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5


def normalize(img, img_mean, img_scale):
    """
    all numpy functions are tensor/vector operations i.e. operations are broadcasted
    """
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def preprocess_image(img):
    """
    1. normalize and tranpose (i.e. switch the dimensions to (720,1280,3) (height,width,channels)
    2. convert to the correct tensor shape and data type
    """
    img = cv2.resize(img, (640,360))
    img = normalize(img, [128,128,128] ,1/256).transpose(2,0,1)
    return np.expand_dims(img,0).astype(np.float32)



def find_peaks(heatmap):
    """
    Given a 2D heatmap, return a list of (x,y) positions of all local maxima (peaks)
    """
    # clean up heatmap by setting all values < 0.3 to 0
    heatmap[heatmap < 0.2] = 0
    
    # simple nested loop to check for peaks (boundary pixels are ignored)
    h,w = heatmap.shape
    peak_locations = []
    for i in range(h-2):
        i+=1
        for j in range(w-2):
            j+=1
            peak = (heatmap[i][j] > heatmap[i-1][j]) & (heatmap[i][j] > heatmap[i+1][j]) \
                    & (heatmap[i][j] > heatmap[i][j-1]) & (heatmap[i][j] > heatmap[i][j+1])
            if peak:
                peak_locations.append((i/h,j/w)) # positions are normalized from 0-1 so that it's not affected by changes in image size (for now it's fixed at 360*640)
    return peak_locations

    
    
def choose_best_peak(peak_locations, calib):
    """
    Find the peak in peak_locations that is the closest to some calibration location  
    """
    min_distance = 99999
    best_idx = 0
    for i, peak_location in enumerate(peak_locations):
        distance = find_distance(peak_location, calib)
        if distance < min_distance:
            min_distance = distance
            best_idx = i
    return peak_locations[best_idx]

    
    
def get_keypoint_locations(output_tensor, poi, calib):
    keypoint_locations = []
    num_bodyparts = 0

    for i, kpt in enumerate(poi):
        heatmap = np.squeeze(output_tensor[0][:,kpt,:,:]) # convert it back to 2D array
        peak_locations = find_peaks(heatmap) # get peak locations
        
        if len(peak_locations): # check that there are peaks
            best_peak = choose_best_peak(peak_locations, calib[i]) 
            keypoint_locations.append((best_peak[0],best_peak[1]))
            num_bodyparts +=1 # count how many peaks are detected

    if num_bodyparts != len(poi):
        # if less than the required number of keypoints, 6, then no data will be collected"""
        print("{} Keypoints missing!".format(6-num_bodyparts))
        pass
    else:
        return keypoint_locations

    
    
def get_signals_from_keypoints(keypoint_locations):
    """
    This processes the keypoints to obtain signals such as neck angle etc.
    Inputs:
    - keypoints      x and y positions of keypoints in the image frame

    Outputs (signals):
    - sho_angle      shoulder angle measured in degrees relative to the horizon
    - sho_distance   shoulder distance in arbitrary unit (since actual distance depends on
                     camera's focal length and user's shoulder width, which are both unknown)
    - sho_y          shoulder y-position in image frame (the average of left & right shoulders)
    - neck_angle     angle in degrees of baseOfNeck-to-nose line to the line perpendicular to the shoulders
    - eye_distance   eye-to-camera distance in arbitrary unit
    - head_pitch     based on how low the tip of the nose is, relative to the eyes
    """
    nose = keypoint_locations[0]
    neck = keypoint_locations[1]
    r_sho = keypoint_locations[2]
    l_sho = keypoint_locations[3]

    sho_angle = np.arctan((r_sho[0] - l_sho[0])/(r_sho[1] - l_sho[1]))
    sho_angle *= (180/np.pi)

    sho_width = find_distance(r_sho, l_sho)
    sho_distance = 100/sho_width

    sho_y = np.mean([neck[0], r_sho[0], l_sho[0]])

    neck_len = find_distance(keypoint_locations[0], keypoint_locations[1])
    abs_neck_angle = np.arctan((nose[1] - neck[1])/(nose[0] - neck[0]))
    abs_neck_angle *= (180/np.pi)

    neck_angle = abs_neck_angle + sho_angle

    r_eye, l_eye = keypoint_locations[4], keypoint_locations[5]
    interocular_distance = find_distance(r_eye, l_eye)
    eye_distance = 10/interocular_distance
    return sho_angle, sho_distance, sho_y, neck_angle, eye_distance



def get_scores_from_signals(frame_signals, calib_signals): 
    """
    calib_signals - signals from calibration keypoint positions, as defined initially
    frame_signals - signals from keypoint positions of current frame
    
    """
    sho_square_score = sigmoid(-1,3,4, 1.047, abs(calib_signals[0] - frame_signals[0]))
    neck_angle_score = sigmoid(-1,3,4, 1.047, abs(calib_signals[3] - frame_signals[3]))

    change_in_eyescreen_distance = frame_signals[4]/calib_signals[4] - 1

    if change_in_eyescreen_distance < 0:
        eyescreen_distance_score = sigmoid(-1,3,9, 1.047, abs(change_in_eyescreen_distance)*100) 
    else:
        eyescreen_distance_score = 1

    change_in_shoulder_distance = np.min([0,frame_signals[1]/calib_signals[1] - 1]) # % change, only capture negative
    slouch_multiplier = sigmoid(1,3,1,0, abs(change_in_shoulder_distance)*100)

    change_in_shoulder_y_norm = (frame_signals[2] - calib_signals[2])/(100/(calib_signals[1]))
    if change_in_shoulder_y_norm > 0:
        slouch_score = 1 - slouch_multiplier * sigmoid(1,3,2,0,change_in_shoulder_y_norm*100)
    else:
        slouch_score = 1
        
    return [sho_square_score, neck_angle_score, eyescreen_distance_score, slouch_score]