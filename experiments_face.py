# Import the necessary modules for this script
import cv2
import extras.face_crop as cropper

# Load the image using OpenCV's `imread` function
img = cv2.imread('lena.png')

# Crop the image using the `crop_image` function from the `face_crop` module
result = cropper.crop_image(img)

# Save the cropped image to a new file using OpenCV's `imwrite` function
cv2.imwrite('lena_result.png', result)
