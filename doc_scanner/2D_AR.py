import os
import cv2
import numpy as np
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('-image', help = 'give the path of input image')
# parser.add_argument('-output', help = 'give path of the folder to save output image')
# args = parser.parse_args()





def projective_transform(paper_img, overlay_img):

	# converting to grayscale since the threshloding requires grayscale image

	gray_paper_img = cv2.cvtColor(paper_img, cv2.COLOR_BGR2GRAY)

	# gaussian blur
	blur_img = cv2.GaussianBlur(gray_paper_img, (3, 3), 100)

	#otsu's thresholding
	_, thresh_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	# using imge morphology
	kernel = np.ones((7, 7), np.uint8)
	morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
	# morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, kernel) # try uncommenting this if only closing doesn't work


	# find largest contour
	contours = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]

	area_thresh = 0
	for c in contours:
		area = cv2.contourArea(c)
		if area > area_thresh:
			area_thresh = area
			longest_contour = c

	# visualizing
	page = np.zeros_like(paper_img)
	cv2.drawContours(page, [longest_contour], 0, (255, 255, 255), -1)

	# get perimeter and approximate a polygon
	peri = cv2.arcLength(longest_contour, True)
	corners = cv2.approxPolyDP(longest_contour, 0.04 * peri, True)

	cor_x = []
	cor_y = []
	for corner in corners:
		pt = [corner[0][0], corner[0][1]]
		cor_x.append(corner[0][0])
		cor_y.append(corner[0][1])
	
	cor_x = sorted(cor_x)
	cor_y = sorted(cor_y)

	# print(cor_x)
	# print(cor_y)

	icorners = [[], [], [], []]
	for corner in corners:
		pt = [corner[0][0], corner[0][1]]
		# print(pt)
		if corner[0][0] in cor_x[2:]:
			if corner[0][1] in cor_y[2:]:
				icorners[3] = pt
			else:
				icorners[2] = pt
		else:
			if corner[0][1] in cor_y[2:]:
				icorners[1] = pt
			else:
				icorners[0] = pt
	# print(icorners)

	icorners = np.float32(icorners)
	h, w, c = overlay_img.shape
	ocorners = [[0, 0], [0, h], [w, 0], [w, h]]
	ocorners = np.float32(ocorners)

	matrix = cv2.getPerspectiveTransform(ocorners, icorners)
	warped = cv2.warpPerspective(paper_img, matrix, (300, 500))
	persp_img = cv2.warpPerspective(overlay_img, matrix, (paper_img.shape[1], paper_img.shape[0]))
	cv2.copyTo(src = persp_img, mask = np.tile(persp_img, 1), dst = paper_img)

	return paper_img, corners




paper = cv2.imread('paper1.jpeg')
overlay_img = cv2.imread('overlay.jpeg')

final_img, corners = projective_transform(paper, overlay_img)
print(corners)


cv2.imshow('output', paper)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('output', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('results' + os.sep + 'result.jpg', final_img)








# video = cv2.VideoCapture('/Users/utsavmdesai/Documents/Coding/CV/video.mp4')

# frameSize = (500, 500)
# out = cv2.VideoWriter('/Users/utsavmdesai/Documents/Coding/CV/results/output_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)


# i = 0
# while(video.isOpened()):
# 	i += 1
# 	ret, frame = video.read()
# 	modified_frame, corners = projective_transform(frame, overlay_img)
# 	out.write(modified_frame)
# 	if ret == True:
# 		cv2.imshow('Frame', modified_frame)

# 		if cv2.waitKey(25) & 0xFF == ord('q'):
# 			break
# 	else:
# 		break
# video.release()
# cv2.destroyAllWindows()
# print(i)