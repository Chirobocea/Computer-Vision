import cv2 as cv

def show_image(title,image,k):
    image=cv.resize(image,(0,0),fx=k,fy=k)
    cv.imshow(title,image)
    cv.waitKey(0)   
    cv.destroyAllWindows()

def show_board(lines_horizontal, lines_vertical, board):
    for line in  lines_vertical : 
        cv.line(board, line[0], line[1], (0, 255, 0), 5)
    for line in  lines_horizontal : 
        cv.line(board, line[0], line[1], (0, 0, 255), 5)
    show_image('Board', board, 0.8)

def show_configuration(board,matrix,lines_horizontal,lines_vertical):
    config = board.copy()
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + 5
            y_max = lines_vertical[j + 1][1][0] - 5
            x_min = lines_horizontal[i][0][1] + 5
            x_max = lines_horizontal[i + 1][1][1] - 5
            if matrix[i][j] == '+':
                config = cv.rectangle(config, (y_min, x_min), (y_max, x_max), color=(255, 0, 0), thickness=3)
    show_image("Configuration", config, 0.8)
        