# crop out isolated pieces from the contours and save them as separate images
import cv2
import numpy as np

def crop_pieces(image, contours):
    """
    This function takes an image and a list of contours as input and crops out the pieces from the image based on the contours.
    :param image: The input image containing the original total puzzle pieces.
    :param contours: A list of contours representing the segmented pieces.
    :return: A list of cropped piece images.
    """
    cropped_pieces = []

    for i, contour in enumerate(contours):
        # get the bounding rectangle coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # crop the piece from the original image using the bounding rectangle coordinates
        cropped_piece = image[y:y+h, x:x+w]
        cropped_pieces.append(cropped_piece)

    return cropped_pieces
