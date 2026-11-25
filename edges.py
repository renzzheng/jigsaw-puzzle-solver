# edge extraction step
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

def compute_angles(pts):
    n = len(pts)
    angles = []
    for i in range(n):
        A = pts[i - 1]        # previous point (wraps around)
        B = pts[i]            # current point
        C = pts[(i + 1) % n]  # next point (wraps around)
        
        BA = A - B
        BC = C - B
        
        norm = np.linalg.norm(BA) * np.linalg.norm(BC)
        if norm == 0:
            angles.append(0)
            continue
            
        cos_angle = np.dot(BA, BC) / norm
        cos_angle = np.clip(cos_angle, -1, 1)  # prevent arccos errors
        angles.append(np.degrees(np.arccos(cos_angle)))
    
    return np.array(angles)

def split_edges(pts, corner_indices):
    """
    This function takes the contour points and the corner indices as input and splits the contour into edges based on the corner indices.
    :param pts: The contour points representing a segmented piece.
    :param corner_indices: A list of indices representing the corners of the puzzle piece.
    :return: A list of edges, where each edge is a list of points between two corners.
    """
    edges = []
    n = len(corner_indices)
    
    for i in range(n):
        start_idx = corner_indices[i]
        end_idx = corner_indices[(i + 1) % n]  # wrap around to the first corner
        if start_idx < end_idx:
            edge = pts[start_idx:end_idx + 1]  # include end point
        else:
            edge = np.concatenate((pts[start_idx:], pts[:end_idx + 1]), axis=0)  # wrap around edge
        edges.append(edge)
    
    return edges

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

    angles = compute_angles(pts)
    angles = gaussian_filter(angles, sigma=3)

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

    # convert peak indices back to original contour indexing
    corner_indices = (peaks + start) % len(pts)
    corner_indices = np.sort(corner_indices)

    edges = split_edges(pts, corner_indices)

    return edges