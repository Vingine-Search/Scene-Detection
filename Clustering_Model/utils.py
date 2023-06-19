import numpy as np
import cv2
def mean_pixel_distance(left: np.ndarray, right: np.ndarray) -> float:
    assert len(left.shape) == 2 and len(right.shape) == 2
    assert left.shape == right.shape
    num_pixels: float = float(left.shape[0] * left.shape[1])
    return (np.sum(np.abs(left.astype(np.int32) - right.astype(np.int32))) / num_pixels)

def hist_compare(hist_fram1: np.ndarray, hist_fram2: np.ndarray) -> float:
    metric_val: float =cv2.compareHist(hist_fram1, hist_fram2, cv2.HISTCMP_CORREL)
    return metric_val


