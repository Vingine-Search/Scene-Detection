# imports
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from operator import itemgetter
import cv2
import numpy as np

from dataclasses import dataclass,fields
from utils import *
from typing import List, Tuple

def get_distance_matrix(features_matrix):
    D_Matrix = sparse.csr_matrix(features_matrix)
    D_Matrix = cosine_similarity(D_Matrix)
    D_Matrix = 1-D_Matrix
    # D_Matrix.tolist()
    return D_Matrix
#TODO: try eculedian  distance 

"""
params : 
boundary_frames : frames classified to be boundaries thsi is list of tuples (data, frame index,hsv_hist)
return : 2D distance matrix (Matrix 2D with size N*N ,N is the numer of shots,cell (i,j) have the feature distance between i,j shots)
"""
def get_distance_matrix_hist(boundary_frames):
    num_shots = len(boundary_frames)
    dist_matrix = np.zeros((num_shots, num_shots))
    for i in range(num_shots):
        print("i",i)
        for j in range(i, num_shots):
            if i == j:
                dist_matrix[i, j] = 0
            else:
                # compute the distance between the i-th and j-th shots' histograms
                dist: float =cv2.compareHist(boundary_frames[i][0].hsv_hist, boundary_frames[j][0].hsv_hist, cv2.HISTCMP_CORREL)
                print(dist)
                dist_matrix[i, j] = 1.0- dist
                dist_matrix[j, i] = 1.0- dist
    return dist_matrix

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


def get_optimal_sequence_addition_cost(D_matrix, scenes_num):
    shots_num = D_matrix.shape[0]
    if shots_num < 1 or scenes_num < 1 or shots_num != D_matrix.shape[1]:
        print("Error: There is an error in shots or scenes size")
        return []
    # check if the size of scenes < shots return shot boundaries 
    if scenes_num > shots_num:
        return np.arrange(1, shots_num + 1)
    if scenes_num == 1:
        return [shots_num - 1]
    D_sum = get_internal_sums(D_matrix,shots_num)
    cost_matrix = np.zeros((shots_num, scenes_num+1))
    index_boundary_matrix = np.zeros((shots_num, scenes_num+1))
    # initialization
    for n in range(0, shots_num):
        cost_matrix[n, 1] = D_sum[n, shots_num - 1]
        index_boundary_matrix[n, 1] = shots_num - 1

    for k in range(2, scenes_num+1):
        for n in range(0, shots_num - k):
            temp = np.transpose(D_sum[n, n:shots_num- k]) + cost_matrix[n + 1:shots_num - k + 1, k - 1]
            index_boundary_matrix[n, k] = np.argmin(temp)
            cost_matrix[n, k] = temp[int(index_boundary_matrix[n, k])]
            index_boundary_matrix[n, k] = index_boundary_matrix[n, k] + n
    boundary_frame_index = np.zeros(scenes_num)
    the_prev = -1

    for k in range(1, scenes_num+1):
        boundary_frame_index[k] = index_boundary_matrix[int(the_prev + 1), scenes_num - k - 1]
        the_prev = boundary_frame_index[k]
    return boundary_frame_index


def get_optimal_sequence_norm_cost(D_matrix, scenes_num):
    return 
