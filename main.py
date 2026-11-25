# main script
import cv2
import os, sys
import segmentation
import pieces
import edges
import matching
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

all_piece_edges = []

for i, piece in enumerate(cropped_pieces):
    cv2.imshow(f"Piece {i}", piece)

    piece_edges = edges.extract_edges(contours[i])
    all_piece_edges.append(piece_edges)

    print(f"{len(piece_edges)} edges found for piece {i}")
    for j, edge in enumerate(piece_edges):
        print(f"  Edge {j}: {len(edge)} points")

all_matches = []

for i in range(len(all_piece_edges)):
    for j in range(i + 1, len(all_piece_edges)):
        for edge_i_idx, edge_i in enumerate(all_piece_edges[i]):
            for edge_j_idx, edge_j in enumerate(all_piece_edges[j]):
                is_match, score = matching.match_edges(edge_i, edge_j)

                if is_match:
                    all_matches.append(
                        (score, i, edge_i_idx, j, edge_j_idx, edge_i, edge_j)
                    )

all_matches.sort(key=lambda x: x[0])

best_match_for_edge = {}

for score, i, edge_i_idx, j, edge_j_idx, edge_i, edge_j in all_matches:
    key1 = (i, edge_i_idx)
    key2 = (j, edge_j_idx)

    if key1 not in best_match_for_edge:
        best_match_for_edge[key1] = (score, j, edge_j_idx, edge_i, edge_j)

    if key2 not in best_match_for_edge:
        best_match_for_edge[key2] = (score, i, edge_i_idx, edge_j, edge_i)

seen_pairs = set()
mutual_matches = []

for key_a, match_a in best_match_for_edge.items():
    piece_a, edge_a_idx = key_a
    score_a, piece_b, edge_b_idx, edge_a, edge_b = match_a

    key_b = (piece_b, edge_b_idx)

    if key_b not in best_match_for_edge:
        continue

    score_b, back_piece, back_edge_idx, _, _ = best_match_for_edge[key_b]

    # Keep only matches where A's best is B and B's best is A
    if back_piece == piece_a and back_edge_idx == edge_a_idx:
        pair_key = frozenset([key_a, key_b])

        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            mutual_matches.append((score_a, key_a, key_b, edge_a, edge_b))

mutual_matches.sort(key=lambda x: x[0])

print("\nMutual best matches:")

for score, key_a, key_b, edge_a, edge_b in mutual_matches[:10]:
    piece_a, edge_a_idx = key_a
    piece_b, edge_b_idx = key_b

    print(
        f"Piece {piece_a} edge {edge_a_idx} ↔ "
        f"Piece {piece_b} edge {edge_b_idx}. Score: {score:.4f}"
    )

    matching.plot_edge_match(edge_a, edge_b)

cv2.waitKey(0)
cv2.destroyAllWindows()