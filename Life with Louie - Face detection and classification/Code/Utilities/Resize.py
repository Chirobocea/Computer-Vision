import cv2
import os

def resize(image, size=224):
    resized_image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    print("resized")
    return resized_image

image_dir_faces = "D:/Cursuri/5. Computer Vision/Proiect 2/Task 2"
image_save = "D:/Cursuri/5. Computer Vision/Proiect 2/Task 2/Resized/Validation"

for filename in os.listdir(image_dir_faces):

    if filename.endswith('.jpg'):

        image = cv2.imread(os.path.join(image_dir_faces, filename))

        image = resize(image)

        image_path = os.path.join(image_save, filename)
        cv2.imwrite(image_path, image)