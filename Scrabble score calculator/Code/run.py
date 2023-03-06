import cv2 as cv
import os

from task1 import *
from task2 import *
from task3 import *
from visualisation import *
from utilites import *

def run(images_location, mask_location, solutions_location):

    files=os.listdir(images_location)
    mean_sat = get_mean_saturation_value(files, images_location)
    
    i=1
    j=1
    last_matrix = None
    for file in files:
        if file[-3:]=='jpg':
            if j >20:
                i = i+1
                j = 1
                last_matrix = None

            image = cv.imread(images_location+'/'+file) 
            board = extract_board(image, mean_sat)
            lines_horizontal, lines_vertical = make_lines(900, 60)
            current_matrix = board_configuration(board, lines_horizontal, lines_vertical)
            diff_matrix = find_new_changes(last_matrix, current_matrix)
            new_letters, str_config = board_configuration_letters(mask_location, board, diff_matrix, lines_horizontal, lines_vertical)
            current_matrix = combine(last_matrix, new_letters)

            last_matrix = current_matrix.copy()

            points = calculate_points(diff_matrix, current_matrix)

            result = str_config+points
            if j<10:
                file_name = str(i)+"_0"+str(j)
            else:
                file_name = str(i)+"_"+str(j)

            file_to_write = open(solutions_location + '/' + file_name+".txt", "w+")
            file_to_write.write(result) 
            file_to_write.close()  

            j = j+1

images_location = 'D:/Cursuri/5. Computer Vision/Proiect 1/CAVA-2022-TEMA1/testare'
mask_location = "C:/Users/chimi/Desktop/Final project/Masks"
solutions_location = "C:/Users/chimi/Desktop/Final project/Solutions"

run(images_location, mask_location, solutions_location)