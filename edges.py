# edge extraction step
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

def extract_edges(contour):
    """
    This function takes a contour as input and extracts the edges of the contour using the Ramer-Douglas-Peucker algorithm to approximate the contour shape.
    :param contour: The input contour representing a segmented piece.
    :return: A list of points representing the approximated edges of the puzzle contour.
    """
    # apply Gaussian smoothing to the contour points to reduce noise and improve edge detection
    pts = contour.reshape(-1, 2).astype(float)
    pts[:, 0] = gaussian_filter(pts[:, 0], sigma=3)
    pts[:, 1] = gaussian_filter(pts[:, 1], sigma=3)

    # # default polygon approximation epsilon
    # pts_contour = pts.reshape(-1, 1, 2).astype(np.float32)
    # epsilon = 0.04 * cv2.arcLength(pts_contour, True)
    # approx = cv2.approxPolyDP(pts_contour, epsilon, True)

    # obtain the moments of the contour to calculate the centroid
    # (1) calculate moments for a specific contour
    M = cv2.moments(pts.reshape(-1, 1, 2).astype(np.float32))
    # (2) solve for centroid coordinates
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0 # handle zero area to avoid division by zero

    # distance from centroid to each point in the approximated contour
    distances = np.sqrt((pts[:, 0] - cX)**2 + (pts[:, 1] - cY)**2) # vectorized distance calculation

    # smooth the distance signal too
    distances = gaussian_filter(distances, sigma=3)

    # rotate signal to start at minimum to avoid wrap-around peak splitting
    start = np.argmin(distances)
    distances = np.roll(distances, -start)

    # find peaks with a minimum prominence
    peaks, properties = find_peaks(distances, prominence=8) # adjust prominence threshold as needed
    sorted_idx = np.argsort(distances[peaks])[::-1]
    peaks = peaks[sorted_idx][:4] # keep only the top 4 peaks

    return distances, peaks