
# imports
from math import ceil
from typing import Set
import tensorflow as tf
import cv2
import os
# from utils import *
from sklearn.metrics.pairwise import cosine_similarity
# imports
import cv2
import os
import numpy as np
import math
from dataclasses import dataclass,fields
# from utils import *
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


def mean_pixel_distance(left: np.ndarray, right: np.ndarray) -> float:
    assert len(left.shape) == 2 and len(right.shape) == 2
    assert left.shape == right.shape
    num_pixels: float = float(left.shape[0] * left.shape[1])
    return (np.sum(np.abs(left.astype(np.int32) - right.astype(np.int32))) / num_pixels)



def hist_compare(hist_fram1: np.ndarray, hist_fram2: np.ndarray) -> float:
    metric_val: float =cv2.compareHist(hist_fram1, hist_fram2, cv2.HISTCMP_CORREL)
    return metric_val
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


"""get the total sum of distance of different segmentation 
params :
 Distance_Matrix: 2D distance matrix 
 N: number of shots 
return : table with all different segments in D_Matrix
"""
def get_internal_sums(D_Matrix,N):
    # D_sum = torch.zeros(N,N, device=self.device)
    D_sum = np.zeros(shape=(N,N))
    # initialize diagonal 
    for shot_index in range(N):
        D_sum[shot_index,shot_index] = D_Matrix[shot_index,shot_index]
    # # initialize second diagonal
    # for shot_index in range(0, N-1):
    #     D_sum[shot_index, shot_index+1] = D_Matrix[shot_index:shot_index+1+1, shot_index:shot_index+1+1].sum()
    #     print(D_sum[shot_index, shot_index+1])
    #     # TODO: recheck this condition 
    #     D_sum[shot_index+1, shot_index] = D_sum[shot_index, shot_index+1]
    #     print(D_sum[shot_index+1, shot_index])
    #TODO: if you change the 1 to 2 you should discomment the above code 
    for scene_size in range(1, N):
        for start_shot in range(0, N - scene_size):
            '''
            D_sum[i,j] =+D_sum[i-1,j]                  --> missing last row
                        +D_sum[i,j-1]                  --> missing last column 
                        -D_sum[i-1,j-1]                --> as we sum it twice before
                        +D_Matrix[i,j] + D_Matrix[j,i] -->missing cells in brevious calculations  
            '''
            D_sum[start_shot, start_shot + scene_size] = D_sum[start_shot, start_shot + scene_size - 1] + D_sum[start_shot - 1, start_shot + scene_size] - D_sum[start_shot - 1, start_shot + scene_size - 1] +  D_Matrix[start_shot, start_shot + scene_size] + D_Matrix[start_shot + scene_size, start_shot] 
            # as the matrix is symetric 
            D_sum[start_shot + scene_size, start_shot] = D_sum[start_shot, start_shot + scene_size]
    return D_sum
#################################################################
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
        return (None,video_param.last_frame,hist_def)
    print("process")
    ##################################################################################################
    # try use the histograme of the boundary not the avg of the frames 
    # shot_score = op(frames_since_last_cut) if frame_score >= video_param.threshold else None
    shot_score = frame_score if frame_score >= video_param.threshold else None
    print(shot_score)
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

import shutil
def get_framesboundary_data(video_path="./Dataset/tears_of_steel_1080p.mov",output_dir="./Dataset/frames",output_dir_shot_boundries = "./Dataset/shot_boundary",sampling_rate=30):
    boundary_frames: List[frame_data] = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(output_dir_shot_boundries):
        shutil.rmtree(output_dir_shot_boundries)
    os.makedirs(output_dir_shot_boundries)
    cap = cv2.VideoCapture(video_path)
    video_try = shot_detector()
    # Get the frame rate.
    fps = cap.get(cv2.CAP_PROP_FPS)
    # assert video_try.min_scene_len > sampling_rate, "The sampling rate is must be strictly less than the minimum scene length"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get the video duration in seconds 
    seconds = round(total_frames / fps)
    # 2 min
    if seconds < 120:
        sampling_rate = 10
    # 3 min
    elif seconds < 180:
        sampling_rate = 20
    # 4 min
    elif seconds < 240:
        sampling_rate = 30
    else:
        sampling_rate = 60

    frame_data_since_last_cut = []
    for frame_idx in range(0, total_frames, sampling_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, "frame_{:02d}_{:06d}.jpg".format(int(frame_idx // fps), frame_idx))
            cv2.imwrite(output_path, frame)
        shot_score, data,hist_def = process_frame(
            video_try, frame_data_since_last_cut,
            len(frame_data_since_last_cut),
            frame, avg_frames_features)
        if shot_score is not None:
            print(frame_idx)
            output_path = os.path.join(output_dir_shot_boundries, "frame_{:02d}_{:06d}.jpg".format(int(frame_idx//fps), frame_idx))
            cv2.imwrite(output_path, frame)
            # Reset the data accumulator.
            frame_data_since_last_cut = []
            boundary_frames.append((data,frame_idx,frame_idx//fps))
        frame_data_since_last_cut.append(data)
        ######################## USING HISTOGRAM ###################################
        # if hist_def < video_try.threshold_hist:
        #     print(frame_idx)
        #     ######################## SAVE SHOT BOUNDARY IN ANOTHER FOLDER TO GET DEEP FEATURES ###################
        #     if not os.path.exists(output_dir_shot_boundries):
        #         os.makedirs(output_dir_shot_boundries)
        #     output_path = os.path.join(output_dir_shot_boundries, "frame_{:02d}_{:06d}.jpg".format(int(frame_idx//fps), frame_idx))
        #     cv2.imwrite(output_path, frame)
            # boundary_frames.append((data,frame_idx,frame_idx//fps))

    cap.release()
    return boundary_frames
##############################################
def  get_all_possible_sums(e_index_of_big_area,num_small_areas) ->Set[int]:
    if num_small_areas  == 0:
        return {0}
    if num_small_areas  == 1:
        return {e_index_of_big_area**2}
    if e_index_of_big_area < num_small_areas:
        return set()
    possible_sums = set()
    for i in range(ceil(e_index_of_big_area/2)):
        area_i = (i+1)**2
        '''
        try all possible sums  division in the rest of the matrix from i to e_index_of_big_area
        '''
        for remain_s_area  in get_all_possible_sums(e_index_of_big_area-i-1,num_small_areas-1):
            possible_sums.add(area_i+remain_s_area)
    return possible_sums

def DeepFeaturesExtractor():
    inputs = tf.keras.layers.Input((299, 299, 3))
    xcp_preprocessed_inputs = tf.keras.applications.xception.preprocess_input(inputs)
    xcp = tf.keras.applications.Xception(
        input_tensor=xcp_preprocessed_inputs,
        # Don't include the top classification layer, as we will use this model for feature extraction.
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    # For sanity, we shouldn't run `model.fit` anyway.
    xcp.trainable = False
    features = tf.keras.layers.Flatten()(xcp.output)
    return tf.keras.Model(inputs, features)
def get_deep_features_for_key_frames(keyframe_files: list):
    """Returns feature vectors based on `Xception` CNN network for the given list of (key) frames.
    Args:
        keyframe_files: A list of string file names for the frames to extract deep features from.
                        (e.g. key_frames_files=['shot1_keyframe.jpg', 'shot2_keyframe.jpg', ...])
    """
    xcp_feat_ext = DeepFeaturesExtractor()
    image_dimensions = xcp_feat_ext.input.shape[1:]
    # A generator to load the key frames asynchronously.
    def image_generator():
        for keyframe_file in keyframe_files:
            image = tf.image.decode_image(tf.io.read_file(keyframe_file))
            resized_image = tf.image.resize_with_pad(
                image, target_height=image_dimensions[0], target_width=image_dimensions[1])
            yield resized_image
    # Batch every one example in a dataset.
    dataset = tf.data.Dataset.from_generator(
        image_generator, output_shapes=image_dimensions, output_types=(tf.float32)).batch(1)
    return xcp_feat_ext.predict(dataset)
#######################################
def get_optimal_sequence_norm_cost(D_matrix, scenes_num,boundary_frames):
    shots_num = D_matrix.shape[0]
    if shots_num < 1 or scenes_num < 1 or shots_num != D_matrix.shape[1]:
        print("Error: There is an error in shots or scenes size")
        return []
    # check if the size of scenes < shots return shot boundaries 
    if scenes_num > shots_num:
        return [boundary_frames[i][2] for i in range(1,shots_num + 1)]
        # return np.arrange(1, shots_num + 1)
    if scenes_num == 1:
        return [boundary_frames[-1][2]] 
        # return [shots_num - 1]
    
    # D_sum = get_internal_sums(D_matrix,shots_num)
    '''
    every index in the cost_matrix ,index_boundary_matrix ,area_matrix
    '''
    cost_matrix = {}
    index_boundary_matrix = {}
    area_matrix ={}
    # initialization
    for n in range(1, shots_num+1):
        area_n = (shots_num-n+1 )**2
        for remain_a in get_all_possible_sums(n-1,scenes_num-1):
            dist_sum = np.sum(D_matrix[n-1:shots_num,n-1:shots_num])
            index_boundary_matrix[(n,1,remain_a)] = shots_num
            area_matrix[(n,1,remain_a)] = area_n
            cost_matrix[(n,1,remain_a)] = dist_sum / (remain_a+dist_sum)
          
    # the rest of the table
    for k in range(2, scenes_num+1):
        for n in range(1, shots_num - k+1):
            for remain_a in get_all_possible_sums(n-1,scenes_num-k):
                min_cost = np.inf
                min_index = np.inf
                for i in range(n,shots_num-k+1):
                    '''
                    In this for loop we try to get the minimum cost 
                    and its index depending on the the equation of G that 
                    explained in the paper
                    G(n,k,p)(i) = sum(j1=n,i)sum(j2=n,i) D(xj1, xj2) /(p+(i − n + 1)**2+area_matrix(i+1,k-1,p+(i − n + 1)**2))
                    '''
                    area_n_i = (i-n+1)**2
                    a_m = area_matrix.get((i+1,k-1,remain_a+area_n_i),0)
                    G = np.sum(D_matrix[n-1:i,n-1:i])/(remain_a+area_n_i+a_m)
                    c_m =  cost_matrix.get((i + 1, k - 1, remain_a + area_n_i),0)
                    cost = G + c_m
                    if cost < min_cost:
                        min_cost = cost
                        min_index = i
                cost_matrix[(n, k, remain_a)] = min_cost
                index_boundary_matrix[(n, k, remain_a)] = min_index
                p_r = area_matrix.get((min_index + 1, k - 1, remain_a + (min_index - n + 1) ** 2), 0)
                area_matrix[(n, k, remain_a)] = (min_index - n + 1) ** 2 + p_r
    boundary_frame_index = [0]
    boundary_frame_second =[0]
    t_r = 0
    for k in range(1, scenes_num + 1):
        boundary_frame_index.append(index_boundary_matrix[(boundary_frame_index[-1] + 1, scenes_num - k + 1, t_r)])
        boundary_frame_second.append(boundary_frames[boundary_frame_index[k]][2])
        t_r += (boundary_frame_index[-1] - boundary_frame_index[-2]) ** 2
    return np.array(boundary_frame_second[1:]) - 1
########################################
def get_boundary_deep_features_and_normcost(boundary_frames,output_shot_dir='./Dataset/shot_boundary',scenes_num = 5):
    frames_dir = []
    if not os.path.exists(output_shot_dir):
        os.makedirs(output_shot_dir)
    for filename in os.listdir(output_shot_dir):
      file_path = os.path.join(output_shot_dir, filename)
      frames_dir.append(file_path)
    ret = get_deep_features_for_key_frames(frames_dir)
    distance_matrix = 1-cosine_similarity(ret)
    boundary_frame_seconds = get_optimal_sequence_norm_cost(distance_matrix,scenes_num,boundary_frames)
    return boundary_frame_seconds

def get_scene_seg(video):
    boundary_frames = get_framesboundary_data(video,"./Dataset/frames","./Dataset/shot_boundary",60)
    boundary_frame_seconds = get_boundary_deep_features_and_normcost(boundary_frames,'./Dataset/shot_boundary',5)
    return boundary_frame_seconds

boundary_frame_seconds = get_scene_seg("./Dataset/tears_of_steel_1080p.mov")
print(boundary_frame_seconds)

