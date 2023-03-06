import cv2 as cv
import numpy as np

from utilites import *
from visualisation import *

def classify_letter(mask_location, patch, k):
    letters = {
        1: "A",
        2: "B",
        3: "C",
        4: "D",
        5: "E",
        6: "F",
        7: "G",
        8: "H",
        9: "I",
        10: "J",
        11: "L",
        12: "M",
        13: "N",
        14: "O",
        15: "P",
        16: "R",
        17: "S",
        18: "T",
        19: "U",
        20: "V",
        21: "X",
        22: "Z",
        23: "?"
    }

    gray_patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)

    maxi=-np.inf
    letter=-1

    for j in range(1, 24):

        img_template=cv.imread(mask_location+'/'+str(j)+'.jpg')
        img_template=cv.cvtColor(img_template, cv.COLOR_BGR2GRAY)
        
        corr = cv.matchTemplate(gray_patch, img_template, cv.TM_CCOEFF_NORMED)
        corr=np.max(corr)
        if corr>maxi:
            maxi=corr
            letter=letters[j]

    return letter

def board_configuration_letters(mask_location,board, matrix, lines_horizontal, lines_vertical):
    col = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
        10: "K",
        11: "L",
        12: "M",
        13: "N",
        14: "O"
    }
    
    k=0

    str_config = ""

    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):

            if matrix[i][j] != '-':
                k = k+1
                patch = cut_patch(board, lines_horizontal, lines_vertical, i, j, 0, 0)
                matrix[i][j] = classify_letter(mask_location, patch, k)

                str_config = str_config + str(i+1)+str(col[j]) + " " + str(matrix[i][j]) + "\n" 

    return matrix, str_config