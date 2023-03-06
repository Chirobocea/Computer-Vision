import os
import cv2

# Directory containing the images
directory = 'D:/Cursuri/5. Computer Vision/Proiect 2/Task 2'
directory2 = 'D:/Cursuri/5. Computer Vision/Proiect 2/Task 2'

# Iterate over all files in the directory
for filename in os.listdir(directory):
  # Check if the file is an image
  if filename.endswith(".jpg"):
    # Construct the full path to the file
    path = os.path.join(directory, filename)

    # Read the image file
    image = cv2.imread(path)

    # Flip the image vertically
    image_flipped = cv2.flip(image, 1)
    print("flip")
    # Save the flipped image
    cv2.imwrite(os.path.join(directory2, filename[:-4]+"_flip.jpg"), image_flipped)
