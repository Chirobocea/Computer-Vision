import cv2 as cv
import numpy as np

from utilites import *
from visualisation import *

def make_mask(image, image_mean_saturation):

    lower = np.array([0, 120, 40], np.uint8)
    upper = np.array([10, 210,160], np.uint8)
    mask_red1 = cv.inRange(image_mean_saturation.copy(), lower, upper)

    lower = np.array([110, 120, 40], np.uint8)
    upper = np.array([180, 210,160], np.uint8)
    mask_red2 = cv.inRange(image_mean_saturation.copy(), lower, upper)

    lower = np.array([75, 80, 180], np.uint8)
    upper = np.array([180, 135, 255], np.uint8)
    mask_white = cv.inRange(image.copy(), lower, upper)

    kernel = np.ones((9, 9), np.uint8)

    mask_red = cv.bitwise_or(mask_red1, mask_red2)

    mask_red = cv.erode(mask_red, kernel)
    mask_red = cv.dilate(mask_red, kernel)

    mask_white = cv.erode(mask_white, kernel)
    mask_white = cv.dilate(mask_white, kernel)

    mask = cv.bitwise_or(mask_red, mask_white)

    return mask

def cut_mask(image, mask):

    W = image.shape[1]
    H = image.shape[0]
    
    class Found(Exception): pass

    try:
        for i in range(H-1, 0, -1):
            for j in range(W):
                if mask[i, j] != 0:
                    mask[i-165:i, :] = 0
                    raise Found
    except Found:
        pass

    try:
        for i in range(H):
            for j in range(W):
                if mask[i, j] != 0:
                    mask[i:i+135, :] =0
                    raise Found
    except Found:
        pass

    return mask

def find_corners(mask):

    contours, _ = cv.findContours(mask,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0

    possible_top_left = None
    possible_bottom_right = None
    possible_top_right = None
    possible_bottom_left = None

    for i in range(len(contours)):
        if(len(contours[i]) > 5):

            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point
                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1]:
                    possible_bottom_right = point
                if possible_top_right is None or point[0] - point[1] > possible_top_right[0] - possible_top_right[1]:
                    possible_top_right = point
                if possible_bottom_left is None or -point[0] + point[1] > -possible_bottom_left[0] + possible_bottom_left[1]:
                    possible_bottom_left = point

            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left
                
    return top_left,top_right,bottom_right,bottom_left

def crop_board(top_left,top_right,bottom_right,bottom_left, width, height, image):

    puzzle = np.array([top_left,top_right,bottom_right,bottom_left], dtype = "float32")
    destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = "float32")

    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)

    image_rgb = cv.cvtColor(image, cv.COLOR_HSV2BGR)
    board = cv.warpPerspective(image_rgb, M, (width, height), flags = cv.BORDER_REFLECT + cv.WARP_FILL_OUTLIERS)

    return board

def extract_board(image, mean_hue):
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    image_mean_saturation = make_mean_saturated(image, mean_hue)
    
    mask = make_mask(image, image_mean_saturation)
    mask = cut_mask(image, mask)

    top_left,top_right,bottom_right,bottom_left = find_corners(mask)
    new_size = 900
    board = crop_board(top_left,top_right,bottom_right,bottom_left, new_size, new_size, image_mean_saturation)

    return board

def board_configuration(board, lines_horizontal, lines_vertical):

    matrix = np.empty((15,15), dtype='str')
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):

            patch = cut_patch(board, lines_horizontal, lines_vertical, i, j, 20, 12)
            patch = cv.resize(patch, (0,0), fx=2, fy=2, interpolation=cv.INTER_NEAREST)
            patch[:,:,2] = cv.medianBlur(patch[:,:,2], 3)

            patch = cv.cvtColor(patch, cv.COLOR_BGR2HSV)

            lower = np.array([100, 0, 0], np.uint8)
            upper = np.array([130, 255, 80], np.uint8)
            mask = cv.inRange(patch.copy(), lower, upper)

            if np.mean(mask) > 0:
                matrix[i][j] = '+'
            else:
                matrix[i][j] = '-'
    return matrix
            