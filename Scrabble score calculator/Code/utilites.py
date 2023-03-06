import cv2 as cv
import numpy as np

def get_mean_saturation_value(files, images_location):
    mean = 0
    n = 0
    for file in files:
        if file[-3:]=='jpg':
            image = cv.imread(images_location+'/'+file)
            image_mean = cv.cvtColor(image,cv.COLOR_BGR2HSV)
            mean = mean + np.mean(image_mean[:,:,1], dtype = 'int32')
            n = n+1
            
    return mean//n

def make_mean_saturated(image, mean_hue):

    mean = np.mean(image[:,:,1], dtype = 'int32')
    
    image_mean_saturation = image.copy()

    """
    saturation_difference = image[:,:,1].copy()
    saturation_difference.fill(mean_hue-mean)
    image_mean_saturation[:,:,1] = cv.add(image[:,:,1], saturation_difference)
    """   

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            new_value = image[i,j,1] - mean + mean_hue

            if new_value > 255:
                image_mean_saturation[i,j,1] = 255
            elif new_value < 0:
                image_mean_saturation[i,j,1] = 0
            else:
                image_mean_saturation[i,j,1] = new_value

    return image_mean_saturation            

def make_lines(n, k):
    lines_horizontal=[]
    for i in range(0,n+1,k):
        l=[]
        l.append((0,i))
        l.append((n,i))
        lines_horizontal.append(l)

    lines_vertical=[]
    for i in range(0,n+1,k):
        l=[]
        l.append((i,0))
        l.append((i,n))
        lines_vertical.append(l)

    return lines_horizontal, lines_vertical

def cut_patch(image, lines_horizontal, lines_vertical, i, j, err_horizontal, err_vertical):
    y_min = lines_vertical[j][0][0] + err_vertical
    y_max = lines_vertical[j + 1][1][0] - err_vertical
    x_min = lines_horizontal[i][0][1] + err_horizontal
    x_max = lines_horizontal[i + 1][1][1] - err_horizontal
    patch = image[x_min:x_max, y_min:y_max].copy()
    return patch

def find_new_changes(last_matrix, current_matrix):
    if last_matrix is None:
        return current_matrix.copy()
    else:
        diff = np.empty_like(last_matrix)
        for i in range(last_matrix.shape[0]):
            for j in range(last_matrix.shape[1]):
                if last_matrix[i][j] == '-' and current_matrix[i][j] != '-':
                    diff[i][j] = current_matrix[i][j]
                else:
                    diff[i][j] = '-'
        return diff

def combine(last_matrix, current_matrix):
    if last_matrix is None:
        return current_matrix.copy()
    else:
        comb = current_matrix.copy()
        for i in range(last_matrix.shape[0]):
            for j in range(last_matrix.shape[1]):
                if last_matrix[i][j] != '-':
                    comb[i][j] = last_matrix[i][j]

        return comb