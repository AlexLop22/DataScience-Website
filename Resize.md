-Resize Image

import matplotlib.pyplot as plt
import numpy as np 
from skimage.io import imread
from skimage.transform import resize

url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Claude_Monet%2C_Saint-Georges_majeur_au_cr%C3%A9puscule.jpg/800px-Claude_Monet%2C_Saint-Georges_majeur_au_cr%C3%A9puscule.jpg'
im = imread(url)
plt.imshow(im);
im = resize(im,(512,512))
plt.imshow(im);

-Using scikit-image

from skimage import io, transform

# Load the image
img = io.imread('path_to_image.jpg')

# Resize the image
new_width, new_height = 100, 100  # Set the desired dimensions
resized_img = transform.resize(img, (new_width, new_height), mode='constant')

# Save the resized image
io.imsave('resized_image.jpg', resized_img)

-Using OpenCV:

import cv2

# Load the image
img = cv2.imread('path_to_image.jpg')

# Resize the image
new_width, new_height = 100, 100  # Set the desired dimensions
resized_img = cv2.resize(img, (new_width, new_height))

# Save the resized image
cv2.imwrite('resized_image.jpg', resized_img)
