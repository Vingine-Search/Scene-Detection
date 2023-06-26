from .Extract_Features.visual import get_framesboundary_data
from .Clustering_Model.Scene_Clustering_Algo import get_distance_matrix_hist,get_optimal_sequence_addition_cost
from .Extract_Features.deep_features import get_deep_features_for_key_frames
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os
def get_boundary_hsv_and_additioncost(frames_boundary,scenes_num = 5):
    distance_matrix = get_distance_matrix_hist(frames_boundary)
    boundary_locations = get_optimal_sequence_addition_cost(distance_matrix,scenes_num)
    return boundary_locations


def get_boundary_deep_features_and_additioncost(output_shot_dir='./Dataset/shot_boundary',scenes_num = 5):
    frames_dir = []
    if not os.path.exists(output_shot_dir):
        os.makedirs(output_shot_dir)
    for filename in os.listdir(output_shot_dir):
      file_path = os.path.join(output_shot_dir, filename)
      frames_dir.append(file_path)
    ret = get_deep_features_for_key_frames(frames_dir)
    distance_matrix = 1-cosine_similarity(ret)
    boundary_locations = get_optimal_sequence_addition_cost(distance_matrix,scenes_num)
    return boundary_locations






