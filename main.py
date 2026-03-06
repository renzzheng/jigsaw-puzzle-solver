# main script
import cv2
import sys
import segmentation

img = cv2.imread('imgs/image5.png')
if img is None:
    sys.exit("Error: Image not found or unable to read.")

contours = segmentation.get_individual_pieces(img)

# draw contours on the original image to visualize the segmented pieces
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

cv2.imshow("Displayed Image", img)

print(f"Number of pieces found: {len(contours)}")

# wait for a key press and close the displayed image
k = cv2.waitKey(0)

cv2.destroyAllWindows()
