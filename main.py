# main script
import cv2
import os, sys
import segmentation
import pieces
import edges
import numpy as np
from math import *
# from PIL import Image
from matplotlib import pyplot as plt


img = cv2.imread('imgs/puzzle1.png')
if img is None:
    sys.exit("Error: Image not found or unable to read.")

contours = segmentation.get_individual_pieces(img)

# # isolate valid edges from the contours and filter out noise
# self.canvas = np.zeros((self.gray.shape[0], self.gray.shape[1]), dtype=np.uint8) + 255

# draw contours on the original image to visualize the segmented pieces
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
cv2.imshow("Displayed Image", img)

print(f"Number of pieces found: {len(contours)}")

cropped_pieces = pieces.crop_pieces(img, contours)
for i, piece in enumerate(cropped_pieces):
    cv2.imshow(f"Piece {i}", piece)

    # puzzle_edges = edges.extract_edges(contours[i])
    # for j, edge in enumerate(puzzle_edges):
    #     print(f"Edge {i}: {edge}")

    # wait for a key press and close the displayed image

    distances, peaks = edges.extract_edges(contours[i])
    print(f"Distances for piece {i}: {distances}")
    plt.plot(distances)
    plt.plot(peaks, distances[peaks], "x", color="red")
    plt.title(f"Distance from centroid for piece {i}")
    plt.xlabel("Contour Point Index")
    plt.ylabel("Distance")
    # plt.show()

    k = cv2.waitKey(0)

cv2.destroyAllWindows()
