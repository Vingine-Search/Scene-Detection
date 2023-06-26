import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def mean_pixel_distance(left: np.ndarray, right: np.ndarray) -> float:
    assert len(left.shape) == 2 and len(right.shape) == 2
    assert left.shape == right.shape
    num_pixels: float = float(left.shape[0] * left.shape[1])
    return (np.sum(np.abs(left.astype(np.int32) - right.astype(np.int32))) / num_pixels)

def hist_compare(hist_fram1: np.ndarray, hist_fram2: np.ndarray) -> float:
    metric_val: float =cv2.compareHist(hist_fram1, hist_fram2, cv2.HISTCMP_CORREL)
    return metric_val

def cosine_dist(shot_vectors):
    sim = cosine_similarity([shot_vectors])
    return sim 

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


def get_optimal_sequence_add2(D_matrix, scenes_num):

        shots_num = D_matrix.shape[0]

        if shots_num < 1 or scenes_num < 1 or shots_num != D_matrix.shape[1]:
            print("Error: Problem with input.")
            return []

        if scenes_num > shots_num:
            print("Warning: More scenes than shots. Returning shot boundaries.")
            return np.arrange(1, shots_num + 1)

        if scenes_num == 1:
            return [shots_num - 1]

        D_sum = get_internal_sums(D_matrix,40)

        C = np.zeros((shots_num, scenes_num))
        I = np.zeros((shots_num, scenes_num))

        # initialization
        for nn in range(0, shots_num):
            C[nn, 0] = D_sum[nn, shots_num - 1]
            I[nn, 0] = shots_num - 1

        # fill the rest
        for kk in range(1, scenes_num):
            for nn in range(0, shots_num - kk):
                # T will hold the vector in which we're searching for a minimum
                T = np.transpose(D_sum[nn, nn:shots_num- kk]) + C[nn + 1:shots_num - kk + 1, kk - 1]
                I[nn, kk] = np.argmin(T)
                C[nn, kk] = T[int(I[nn, kk])]
                I[nn, kk] = I[nn, kk] + nn
        # prepare returned boundaries
        boundary_locations = np.zeros(scenes_num)
        the_prev = -1
        for kk in range(0, scenes_num):
            boundary_locations[kk] = I[int(the_prev + 1), scenes_num - kk - 1]
            the_prev = boundary_locations[kk]
        if the_prev != shots_num - 1:
            print("Warning: Encountered an unknown problem.")
        return boundary_locations




