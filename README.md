### Built with:
Python + OpenCV


#### Assumptions
Given a quadrilateral jigsaw puzzle, that is layed out on a white background without any overlap.
> Each piece has 4 distinct corners
> Hence: There should be 4 prominent peaks in the plot

##### Challenges
> Thresholding treating lighter colored pieces as part of the background.
> Resolved by raising threshold to `245` to treat only near-white pixels as background.

> Added Gaussian blur `(5,5)` before thresholding to smooth out paper textures from photo uploads.

> Too many false contours
> Resolved by adding area filtering with a list comprehension, keeping only contours above 1200 pixels

> Canny edge filter or Thresholding ?
> Canny picked up too much internal texture

