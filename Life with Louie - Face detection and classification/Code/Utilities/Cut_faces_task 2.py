import os
import cv2


# The path to the text file with the annotations
annotation_file_path = []
annotation_file_path.append("D:/Cursuri/5. Computer Vision/Proiect 2/CAVA-2022-TEMA2/antrenare/andy_annotations.txt")
annotation_file_path.append("D:/Cursuri/5. Computer Vision/Proiect 2/CAVA-2022-TEMA2/antrenare/louie_annotations.txt")
annotation_file_path.append("D:/Cursuri/5. Computer Vision/Proiect 2/CAVA-2022-TEMA2/antrenare/ora_annotations.txt")
annotation_file_path.append("D:/Cursuri/5. Computer Vision/Proiect 2/CAVA-2022-TEMA2/antrenare/tommy_annotations.txt")

# The path to the directory with the images
image_dir_path = []
image_dir_path.append("D:/Cursuri/5. Computer Vision/Proiect 2/CAVA-2022-TEMA2/antrenare/andy")
image_dir_path.append("D:/Cursuri/5. Computer Vision/Proiect 2/CAVA-2022-TEMA2/antrenare/louie")
image_dir_path.append("D:/Cursuri/5. Computer Vision/Proiect 2/CAVA-2022-TEMA2/antrenare/ora")
image_dir_path.append("D:/Cursuri/5. Computer Vision/Proiect 2/CAVA-2022-TEMA2/antrenare/tommy")

# The path to the directory where the patches will be saved
patch_dir_path = "D:/Cursuri/5. Computer Vision/Proiect 2/Task 2"

# Name for output image
name = ['a', 'l', 'o', 'a']

for i in range(4):
    # Open the text file and read its lines
    with open(annotation_file_path[i], "r") as f:
        lines = f.readlines()

    last_name_file = '-'
    k = 1

    # Iterate through the lines in the text file
    for line in lines:
        # Split the line into its fields
            name_file, x_min, y_min, x_max, y_max, person_name = line.strip().split(" ")

            if person_name == "andy":
                add_name = "andy"
            elif person_name == "louie":
                add_name = "louie"
            elif person_name == "ora":
                add_name = "ora"
            elif person_name == "tommy":
                add_name = "tommy"
            else:
                continue

            if last_name_file == name_file:
                k = k+1 
            else:
                k = 1
            
            # Open the image with the corresponding name
            image_path = os.path.join(image_dir_path[i], name_file)
            image = cv2.imread(image_path)
            
            # Cut the image from (x_min, y_min) to (x_max, y_max)
            cut_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            
            # Save the cut image to the output folder
            output_path = os.path.join(patch_dir_path, add_name+"_"+str(i)+"_"+str(k)+"_" + name_file)
            cv2.imwrite(output_path, cut_image)

            last_name_file = name_file