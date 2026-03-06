# isolate the individual puzzle pieces from the original image
import cv2

def get_individual_pieces(image):
    """
    This function takes an image as input and returns a list of segmented pieces.
    :param image: The input image containing the puzzle pieces.
    :return: A list of segmented pieces.
    """
    # convert the image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply thresholding to create a binary image
    ret, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV) # invert the image to get the pieces as white

    # canny edge detection to find edges in the image (?)

    # find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find external contours only

    return contours
