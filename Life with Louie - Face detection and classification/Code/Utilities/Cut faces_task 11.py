import os
import cv2
import time

# Path to the text file with the information
text_file_path = 'D:/Cursuri/5. Computer Vision/Proiect 2/CAVA-2022-TEMA2/validare/validare_annotations.txt'
# Path to the folder with the images
image_folder_path = 'D:/Cursuri/5. Computer Vision/Proiect 2/CAVA-2022-TEMA2/validare/Validare'
# Path to the folder where the cut images will be saved
output_folder_path = 'D:/Cursuri/5. Computer Vision/Proiect 2/Task 2'

# Start the timer
start_time = time.time()

# Open the text file and read its lines
with open(text_file_path, "r") as f:
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
        image_path = os.path.join(image_folder_path, name_file)
        image = cv2.imread(image_path)
        
        # Cut the image from (x_min, y_min) to (x_max, y_max)
        cut_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        # Save the cut image to the output folder
        output_path = os.path.join(output_folder_path, add_name+"_"+str(k)+"_" + name_file)
        cv2.imwrite(output_path, cut_image)

        last_name_file = name_file

# End the timer
end_time = time.time()

# Calculate the elapsed time in seconds
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
