# imports
import cv2
import os
import numpy as np
import math
from dataclasses import dataclass,fields
from utils import *
from typing import List, Tuple
"""
Define classes 
"""
from typing import Optional
@dataclass
class features_weight:
    # by default set all weights to 1
    dif_weight_hue: float = 1.0
    dif_weight_sat: float = 1.0
    dif_weight_val: float = 1.0
    # TODO: have bigger values that other features,detection threshold may need to be adjusted
    dif_weight_edges: float = 1.0
@dataclass
class frame_data:
    hue: np.ndarray
    sat: np.ndarray
    val: np.ndarray
    edges: np.ndarray
    hsv_hist: Optional[np.ndarray]
    #TODO: add optical flow

# create defualt weights 
features_weight_d = features_weight()
@dataclass
class shot_detector:
    threshold: float = 27.0
    threshold_hist: float = 0.6
    min_scene_len: int = 15
    weights: 'features_weight' = features_weight_d
    kernel_size_edges: Optional[int] = None
    last_frame: Optional[frame_data] = None

"""
desc   : convert_video_to_frames  
params : 1- video_path
         2- output_dir 
return : no return 
"""
def convert_video_to_frames(video_path,output_dir,sampling_rate=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(0, total_frames, sampling_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, "frame_{:06d}.jpg".format(frame_idx))
            cv2.imwrite(output_path, frame)
    cap.release()
    return 
"""
desc   : detect_edges  
params : 1- val : val channel from the HSV image 
           2- kernel : kernel size used in canny 
return : numpy array with dilated edges
"""
import math
def detect_edges(val: np.ndarray,kernel:int =None) -> np.ndarray:
    if kernel == None:
        # calculate kernel size from the video reselution
        kernel_size_edges = 4 + round(math.sqrt(val.shape[1]*val.shape[0]) / 192)
        if kernel_size_edges % 2 == 0:
            kernel_size_edges += 1
        kernel = np.ones((kernel_size_edges, kernel_size_edges), np.uint8) 
    # Estimate levels for thresholding.
    sigma: float = 1.0 / 3.0
    median = np.median(val)
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))
    # TODO : Implement  Canny 
    edges = cv2.Canny(val, low, high)
    return cv2.dilate(edges,kernel)
"""
desc: get_HSV_feature_frame
params : frame
return : feature matrix with 3 histograms for H S V
"""
def get_HSV_feature_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_frame = cv2.calcHist(frame, [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_frame, hist_frame, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist_frame

"""
desc   : calculate score for the frame to detect if it is shot boundry or not  
params : 1- video_param
         2- frame_img 
return : 1- frame_score
         2- hist_def: corelation between the hist of the current and prev frame 
"""
def calculate_frame_score(video_param: shot_detector, frame_img: np.ndarray) -> float:
    hue, sat, val = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))
    # calculate edges 
    edges = detect_edges(val) 
    # calculate histogram for H S V values 
    hist_hsv = get_HSV_feature_frame(frame_img)
    if video_param.last_frame is None:
        video_param.last_frame = frame_data(hue, sat, val, edges,hist_hsv)
        return 0.0,1.0
    # calculate histogram for calcOpticalFlow
    # hist_optical = get_calcOpticalFlow_feature_frame([video_param.last_frame.hue, video_param.last_frame.sat, video_param.last_frame.lum], frame_img)
    #TODO: calculate HOG
    # get score for adjecent frames 
    score_components = features_weight(
        dif_weight_hue=mean_pixel_distance(hue, video_param.last_frame.hue),
        dif_weight_sat=mean_pixel_distance(sat, video_param.last_frame.sat),
        dif_weight_val=mean_pixel_distance(val, video_param.last_frame.val),
        dif_weight_edges=(0.0 if edges is None else mean_pixel_distance(edges, video_param.last_frame.edges)),
    )
    hist_def = hist_compare(hist_hsv,video_param.last_frame.hsv_hist)
    frame_score: float = (
        sum(
            getattr(score_components, field.name) * getattr(video_param.weights, field.name)
            for field in fields(features_weight)
        )
        / sum(abs(getattr(video_param.weights, field.name)) for field in fields(features_weight)))
    # Store all data required to calculate the next frame's score.
    video_param.last_frame = frame_data(hue, sat, val, edges,hist_hsv)
    return frame_score,hist_def
"""
return: List of frames where scene cuts have been detected
"""
def process_frame(
    video_param: shot_detector, frames_since_last_cut: List[frame_data],
    frames_count_since_last_cut: int, frame_img: np.ndarray, op) -> Tuple[Optional[float], frame_data]:
    """Returns the shot representation/embeddings (by performing `op`) for a span of frames
    if `frame_img` is a boundary, None otherwise.
    Also returns the `frame_data` for `frame_img` to be used by the caller.
    """
    frame_score ,hist_def= calculate_frame_score(video_param, frame_img)
    # consider any frame over the threshold a new scene, but only if
    # the minimum scene length has been reached (otherwise it is ignored).
    # NOTE: `frames_count_since_last_cut` is the actual number of frames since the last cut, i.e. not sampled every x frames.

    if frames_count_since_last_cut < video_param.min_scene_len:
        return (None, video_param.last_frame,hist_def)
    ##################################################################################################
    # try use the histograme of the boundary not the avg of the frames 
    shot_score = op(frames_since_last_cut) if frame_score >= video_param.threshold else None
    return (shot_score, video_param.last_frame,hist_def)
def avg_frames_features(frames: List[frame_data]) -> frame_data:
    """A reduction operation to perform (averaging) on a list of frame data.
    The reduced data would then be used as a representation of the whole shot.
    """
    assert frames, "Operation can not be performed on 0 frames"
    avg_frame_data = frame_data(0, 0, 0, None)
    count = len(frames)
    attrs = [f.name for f in fields(frames[0])]
    for attr_name in attrs:
        attr_value = sum(getattr(frame, attr_name) for frame in frames) / count
        setattr(avg_frame_data, attr_name, attr_value)
    return avg_frame_data


def get_framesboundary_data(video_path="./Dataset/tears_of_steel_1080p.mov",output_dir="./Dataset/frames",output_dir_shot_boundries = "./Dataset/shot_boundary",sampling_rate=60):
    boundary_frames: List[frame_data] = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    video_try = shot_detector()
    # Get the frame rate.
    fps = cap.get(cv2.CAP_PROP_FPS)
    # assert video_try.min_scene_len > sampling_rate, "The sampling rate is must be strictly less than the minimum scene length"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_data_since_last_cut = []
    for frame_idx in range(0, total_frames, sampling_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            print(frame_idx // fps)
            output_path = os.path.join(output_dir, "frame_{:02d}_{:06d}.jpg".format(int(frame_idx // fps), frame_idx))
            cv2.imwrite(output_path, frame)
        shot_score, data,hist_def = process_frame(
            video_try, frame_data_since_last_cut,
            len(frame_data_since_last_cut) * sampling_rate,
            frame, avg_frames_features)
        # if shot_score is not None:
        #     print(frame_idx)
        #     # Reset the data accumulator.
        #     frame_data_since_last_cut = []
        # frame_data_since_last_cut.append(data)
        if hist_def < video_try.threshold_hist:
            print(frame_idx)
            ######################## SAVE SHOT BOUNDARY IN ANOTHER FOLDER TO GET DEEP FEATURES ###################
            if not os.path.exists(output_dir_shot_boundries):
                os.makedirs(output_dir_shot_boundries)
            output_path = os.path.join(output_dir_shot_boundries, "frame_{:02d}_{:06d}.jpg".format(int(frame_idx//fps), frame_idx))
            cv2.imwrite(output_path, frame)
            boundary_frames.append((data,frame_idx,frame_idx//fps))
    cap.release()
    return boundary_frames


# get_framesboundary_data()