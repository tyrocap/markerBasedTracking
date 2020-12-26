#!/usr/bin/python3
import cv2 as cv
import argparse
import numpy as np
import math

def findFourPoints(pts):
    """
    A helper func to find 4 angle points from list of contours points
    """
    lowerLeft, upperRight, upperLeft, lowerRight = [100000, 0], [-1, 0], [0, 100000], [0, -1]
    for i in pts:
        upperRight = i[0] if i[0][0] > upperRight[0] else upperRight 
        lowerLeft = i[0] if i[0][0] < lowerLeft[0] else lowerLeft
        lowerRight = i[0] if i[0][1] > lowerRight[1] else lowerRight
        upperLeft = i[0] if i[0][1] < upperLeft[1] else upperLeft
    return [upperRight, lowerLeft, lowerRight, upperLeft]

def filterContours(contours):
    cnts = []
    for i in contours:
        # some attribute values are chosen randomly
        peri = cv.arcLength(i, True)
        approx = cv.approxPolyDP(i, 0.02 * peri, True)
        x, y, w, h = cv.boundingRect(approx)
        # eleminate every contours exCenterept one we need
        if len(approx) == 4 and cv.contourArea(i) > 100 and w//h < 2:
            cnts.append(i)
    return cnts


# input
parser = argparse.ArgumentParser(description="Marked Based Tracking")
parser.add_argument('--input', help='Path to input image.', default='../imgs/markedImg.jpg')
args = parser.parse_args()

# read img
img = cv.imread(args.input)
img = cv.resize(img, (700, 800))

# convert to gray 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# thresholding should be chosen with care;
# TODO: thresholding complex images isn't this simple
# here, we chose 127 for simplicity.
_, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

# find countours & draw them
# TODO: finding contours in complex images isn't this simple
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# filter contours
filteredConts = filterContours(contours)

# find 4 corner points
[upperRight, lowerLeft, lowerRight, upperLeft] = findFourPoints(filteredConts[0])

# find center
xCenter = (upperRight[0] + lowerLeft[0] + lowerRight[0] + upperLeft[0])//4
yCenter = (upperRight[1] + lowerLeft[1] + lowerRight[1] + upperLeft[1])//4


# draw contours
img = cv.drawContours(img, filteredConts, -1, (0, 255, 0), 2)


cv.circle(img, ((upperRight[0] + xCenter)//2, (upperRight[1] + yCenter)//2), 2, (0, 0, 255), 2) # upper right 
cv.circle(img, ((lowerLeft[0] + xCenter)//2, (lowerLeft[1] + yCenter)//2), 2, (0, 0, 255), 2) # lower left 
cv.circle(img, ((lowerRight[0] + xCenter)//2, (lowerRight[1] + yCenter)//2), 2, (0, 0, 255), 2) # lower right 
cv.circle(img, ((upperLeft[0] + xCenter)//2, (upperLeft[1] + yCenter)//2), 2, (0, 0, 255), 2) # upper left

#cv.circle(img, (upperRight[0], upperRight[1]), 10, (0, 0, 255), 2) # upper right 
#cv.circle(img, (lowerLeft[0], lowerLeft[1]), 10, (0, 0, 255), 2) # lower left 
#cv.circle(img, (lowerRight[0], lowerRight[1]), 10, (0, 0, 255), 2) # lower right 
#cv.circle(img, (upperLeft[0], upperLeft[1]), 10, (0, 0, 255), 2) # upper left
#cv.circle(img, (xCenter, yCenter), 10, (0, 255, 0), 2)

cv.imshow("Contoured Img", img)
cv.waitKey(0)
cv.destroyAllWindows()

