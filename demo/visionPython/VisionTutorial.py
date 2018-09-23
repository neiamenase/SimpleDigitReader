## This example is from https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
## Recognizing digits with OpenCV and Python
## Of coz, some minor changes are made by me.
## Thanks to Adrian Rosebrock, I could try to implement this project.

## Noted that # comment is from the article; ## is from me


from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2

# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}
# load the example image
image = cv2.imread("example.jpg")


# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map

## -preprocess
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0) ## blur the image using 5x5 kernal windows. Average the color.
edged = cv2.Canny(blurred, 50, 200, 255)

cv2.imwrite("middle result 01 - edged.jpg", edged) ## middle result

# find contours in the edge map, then sort them by their size in descending order
## From openCV api, contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity.
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
## cnts = set of x-y coordinates
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# loop over the contours
for c in cnts: ## c = each curve (set of points)
	# approximate the contour
	peri = cv2.arcLength(c, True) #	~circumference
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if the contour has four vertices, then we have found
	# the thermostat display
	if len(approx) == 4:  ## Try to locate the smallest 4-edges polygons; personally I do not think its the best way
		displayCnt = approx
		break


# extract the thermostat display, apply a perspective transform
# to it
warped = four_point_transform(gray, displayCnt.reshape(4, 2)) ## ~ transform it into a rectangle
output = four_point_transform(image, displayCnt.reshape(4, 2))

cv2.imwrite("middle result 02 - 4-pt transformed.jpg", warped) ## middle result

# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
## From api, if pixel value is greater than a threshold value, it is assigned one value (may be white), else it is assigned another value (may be black).
##(T, threshImage) = cv2.threshold(src, thresh, maxval, type)
## src = source image; should be in gray scale
## thresh = threshold value used to classify pixel intensities
## maxval = pixel value used if given pixel in the image passes the thresh test.

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))

## trails on kernal size required

## cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
##array([[0, 0, 1, 0, 0],
##       [1, 1, 1, 1, 1],
##       [1, 1, 1, 1, 1],
##       [1, 1, 1, 1, 1],
##       [0, 0, 1, 0, 0]], dtype=uint8)

cv2.imwrite("middle result 03 - thresh.jpg", thresh) ## middle result

thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
## useful in removing noise; if cv2.MORPH_CLOSE, minor holes will be filled
## X     X
##      XXX       X
##      XXX    X
##  X
##    will become
##
##       X
##      XXX
##      XXX

cv2.imwrite("middle result 04 - noise removed.jpg", thresh) ## middle result

# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
digitCnts = []
 
# loop over the digit area candidates
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)
 
	# if the contour is sufficiently large, it must be a digit
	if w >= 15 and (h >= 30 and h <= 40):
		digitCnts.append(c)

output2 = output.copy()
for d in digitCnts:
	for pt in d:
		cv2.circle(output2, (pt[0][0], pt[0][1]), 1, (0,255,255), -1)

cv2.imwrite("middle result 05 - digitCnts.jpg", output2)

# sort the contours from left-to-right, then initialize the
# actual digits themselves
digitCnts = contours.sort_contours(digitCnts,
	method="left-to-right")[0]
digits = []

# loop over each of the digits
for c in digitCnts:
	# extract the digit ROI
	(x, y, w, h) = cv2.boundingRect(c)
	roi = thresh[y:y + h, x:x + w]
 
	# compute the width and height of each of the 7 segments
	# we are going to examine
	(roiH, roiW) = roi.shape
	(dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
	dHC = int(roiH * 0.05)
 
	# define the set of 7 segments
	segments = [
		((0, 0), (w, dH)),	# top
		((0, 0), (dW, h // 2)),	# top-left
		((w - dW, 0), (w, h // 2)),	# top-right
		((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
		((0, h // 2), (dW, h)),	# bottom-left
		((w - dW, h // 2), (w, h)),	# bottom-right
		((0, h - dH), (w, h))	# bottom
	]
	on = [0] * len(segments)

# loop over the segments
	for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
		# extract the segment ROI, count the total number of
		# thresholded pixels in the segment, and then compute
		# the area of the segment
		segROI = roi[yA:yB, xA:xB]
		total = cv2.countNonZero(segROI)
		area = (xB - xA) * (yB - yA)
 
		# if the total number of non-zero pixels is greater than
		# 50% of the area, mark the segment as "on"
		if total / float(area) > 0.5:
			on[i]= 1
 
	# lookup the digit and draw it on the image
	digit = DIGITS_LOOKUP[tuple(on)]
	digits.append(digit)
	cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
	cv2.putText(output, str(digit), (x - 10, y - 10),
	cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# display the digits
print(u"{}{}.{} \u00b0C".format(*digits))
cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)


