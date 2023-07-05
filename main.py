from .Extract_Features.visual import get_framesboundary_data
from .Clustering_Model.Scene_Clustering_Algo import get_distance_matrix_hist,get_optimal_sequence_addition_cost,get_optimal_sequence_norm_cost
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
#############################################################
# import cv2

# # Open the video file
# video_file = './Dataset/tears_of_steel_1080p.mov'
# cap = cv2.VideoCapture(video_file)

# # Get video properties
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# duration = frame_count / fps

# # Set the duration of each smaller video in seconds
# split_duration = 10

# # Compute the number of smaller videos
# num_splits = int(duration / split_duration)

# # Iterate over each smaller video
# for i in range(num_splits):
#     # Compute the start and end frames of the smaller video
#     start_frame = int(i * split_duration * fps)
#     end_frame = int((i+1) * split_duration * fps)

#     # Set the video writer
#     writer = cv2.VideoWriter(f"output_{i}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))

#     # Iterate over each frame in the smaller video
#     for j in range(start_frame, end_frame):
#         # Set the frame position
#         cap.set(cv2.CAP_PROP_POS_FRAMES, j)

#         # Read the frame
#         ret, frame = cap.read()

#         # Write the frame to the output video
#         writer.write(frame)

#     # Release the video writer
#     writer.release()

# # Release the video capture
# cap.release()










