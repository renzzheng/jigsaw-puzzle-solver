# edge extraction step
import cv2
import numpy as np
from scipy.signal import find_peaks

def extract_edges(contour):
    """
    This function takes a contour as input and extracts the edges of the contour using the Ramer-Douglas-Peucker algorithm to approximate the contour shape.
    :param contour: The input contour representing a segmented piece.
    :return: A list of points representing the approximated edges of the puzzle contour.
    """
    # default polygon approximation epsilon
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # obtain the moments of the contour to calculate the centroid
    # (1) calculate moments for a specific contour
    M = cv2.moments(contour)

    # (2) solve for centroid coordinates
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0 # handle zero area to avoid division by zero

    # distance from centroid to each point in the approximated contour
    distances = np.sqrt((contour[:, 0, 0] - cX)**2 + (contour[:, 0, 1] - cY)**2) # vectorized distance calculation

    # rotate signal to start at minimum to avoid wrap-around peak splitting
    start = np.argmin(distances)
    distances = np.roll(distances, -start)

    # find peaks with a minimum prominence
    peaks, properties = find_peaks(distances, prominence=13)

    return distances, peaks