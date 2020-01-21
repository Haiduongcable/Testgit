import cv2
import numpy as np
import imutils
img = cv2.imread('/home/duongnh/Downloads/3.jpg')
binary_map = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
rec,binary_map = cv2.threshold(binary_map,100,255,cv2.THRESH_BINARY)
#resized = imutils.resize(image, width=300)
#ratio = binary_map.shape[0] / float(resized.shape[0])


nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - binary_map, None, None, None, 8, cv2.CV_32S)
# result = np.ones((labels.shape), np.uint8)*255
areas = stats[1:,cv2.CC_STAT_AREA]
#print(areas)
result = np.zeros((labels.shape), np.uint8)
a = 0
for i in range(0, nlabels - 1):
    if areas[i] >= 1000:   #keep
        result[labels == i + 1] = 255
        a = a+1
rec,result_1 = cv2.threshold(result,100,255,cv2.THRESH_BINARY_INV)
#print(np.shape(binary_map))


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 3:
            shape = "triangle"

            # if the shape has 4 vertices, it is either a square or
            # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

            # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

            # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

            # return the name of the shape
        return shape


#image = cv2.imread('/home/duongnh/images/train/threshold_line.jpg')
image = result_1
resized = imutils.resize(image, width=300)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(resized, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
ratio = image.shape[0] / float(resized.shape[0])
cnts = cv2.findContours(binary_map.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)

    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (150,150), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)