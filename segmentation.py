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

    # gaussian blur to reduce noise and improve contour detection
    blurred_img = cv2.GaussianBlur(gray_img, (5,5), 0)

    # apply thresholding to create a binary image
    ret, thresh_img = cv2.threshold(blurred_img, 235, 255, cv2.THRESH_BINARY_INV) # invert the image to get the pieces as white

    # find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find external contours only

    # iterate through the contours and calculate the area of each contour to filter out small noise
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        print(f"Countour {i}: Area = {area}")
    
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 1200]

    return filtered_contours
