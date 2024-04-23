import cv2
import numpy as np
import os

# HSV range
# png0
lower_green = np.array([25, 50, 50])  
upper_green = np.array([85, 255, 255])  


folder_path = '/Users/alina./CODE/EE5003/program_final/HogSvm/test2/2'
outputfolder_path = '/Users/alina./CODE/EE5003/program_final/HogSvm/test2/filter_2'

for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        # Read image
        image = cv2.imread(os.path.join(folder_path, filename))

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a mask for lettuce
        lettuce_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Morphological processing
        kernel = np.ones((5, 5), np.uint8)
        lettuce_mask = cv2.morphologyEx(lettuce_mask, cv2.MORPH_CLOSE, kernel)

        # Extract background region
        background_mask = cv2.bitwise_not(lettuce_mask)

        # Create a white background
        white_background = np.full_like(image, (255, 255, 255), dtype=np.uint8)

        # overlay them onto a white background
        lettuce = cv2.bitwise_and(image, image, mask=lettuce_mask)
        background = cv2.bitwise_and(white_background, white_background, mask=background_mask)
        result = cv2.add(lettuce, background)

        # save image
        output_path = os.path.join(outputfolder_path, 'filter_' + filename)
        cv2.imwrite(output_path, result)

