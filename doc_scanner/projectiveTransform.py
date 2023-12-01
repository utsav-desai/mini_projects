import cv2
import numpy as np

img = cv2.imread('paper1.jpeg')

h, w, c = img.shape

img_coords = np.array([[0, 0], [0, h], [w, 0], [w, h]])

board = cv2.imread('board.jpg')

new_coords = []

def click_event(event, x, y, flags, params):
	if event == cv2.EVENT_LBUTTONDOWN:
		print(x, y)
		cv2.imshow('image', board)
		new_coords.append([x, y])

# click on one corner and press 'Q', repeat this fr all four corners in order: LT, LB, RT, RB
for i in range(4):
	cv2.imshow('test', board)
	cv2.setMouseCallback('test', click_event)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
new_coords = np.array(new_coords)
matrix, _ = cv2.findHomography(img_coords, new_coords, 0)

persp_img = cv2.warpPerspective(img, matrix, (board.shape[1], board.shape[0]))
cv2.imshow('temp', persp_img)
cv2.copyTo(src = persp_img, mask = np.tile(persp_img, 1), dst = board)
cv2.imshow('result', board)
cv2.waitKey(0)
cv2.destroyAllWindows()
