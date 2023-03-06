import os
import random
import cv2
import random

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

# Name for output image
name = ['a', 'l', 'o', 'a']

# The path to the directory containing the images
image_dir_faces = "D:/Cursuri/5. Computer Vision/Proiect 2/Faces/All"

# A list to store the image sizes
sizes = []

# Iterate over the images in the directory
for filename in os.listdir(image_dir_faces):
    # Check if the file has the '.jpg' extension
    if filename.endswith('.jpg'):
        # Load the image
        image = cv2.imread(os.path.join(image_dir_faces, filename))

        # Get the size of the image
        height, width = image.shape[:2]

        if max(height, width) > 50:

            # Add the size to the list
            sizes.append((width, height))


# The path to the directory where the patches will be saved
patch_dir_path = "D:/Cursuri/5. Computer Vision/Proiect 2/Negative-min50"

# Iterating through sizes
no_size = 0
# How many patches of one size we want
batch = 5

# Read the annotation file and create the dictionary
annotation_dict = []

for k in range(4):
    annotation_dict.append({})

    with open(annotation_file_path[k], "r") as annotation_file:
        for line in annotation_file:
            name_file, x_min, y_min, x_max, y_max, person_name = line.strip().split(" ")
            annotation_dict[k][name_file] = annotation_dict[k].get(name_file, [])
            annotation_dict[k][name_file].append([(int(x_min), int(y_min)), (int(x_max), int(y_max))])

for no_size in range(len(sizes)):

    actual_batch_size = 0

    width = sizes[no_size][0]
    height = sizes[no_size][1]

    # Go through each directory

    directories = list(range(4))
    random.shuffle(directories)

    for k in directories:

        # Create a list of tuples from the dictionary
        items = list(annotation_dict[k].items())

        # Shuffle the list of tuples
        random.shuffle(items)
 
        # Go through the dictionary and extract the patches
        for name_file, annotations in items:

            # Open the image
            image_path = os.path.join(image_dir_path[k], name_file)
            image = cv2.imread(image_path)

            # Try 10 times to find a patch in this image
            for j in range(10):

                # Check if it is possible to find a patch
                if image.shape[1] - width < width or image.shape[1] - height < height:
                    break

                # Choose a random top-left corner for the patch
                x = random.randint(0, image.shape[1] - width)
                y = random.randint(0, image.shape[0] - height)

                # Check if the patch intersects with any of the annotations
                intersects = False
                for annotation in annotations:
                    if (x < annotation[1][0] and x + width > annotation[0][0]) and (y < annotation[1][1] and y + height > annotation[0][1]):
                        intersects = True
                        break

                # If the patch intersects with an annotation, try again
                if intersects:
                    continue

                # If the patch does not intersect with any annotation, extract it
                patch = image[y:y+height, x:x+width]
                #annotation_dict[k][name_file].append([(int(x)+width//4, int(y)+height//4), (int(x)+(3*width)//4, int(y)+(3*height)//4)])

                # Save the patch to the patch directory
                patch_path = os.path.join(patch_dir_path, str(no_size+1)+'_'+str(actual_batch_size+1)+'_'+name[k]+"_neg"+name_file[:4]+'.jpg')
                cv2.imwrite(patch_path, patch)

                actual_batch_size = actual_batch_size + 1
                break

            if actual_batch_size == batch:
                break

        if actual_batch_size == batch:
            break