Problem Set 2
Code Link: https://colab.research.google.com/drive/1CyOQ6w9xNwzNpk4RsCWjuVyeRa4x52kb?usp=sharing

Load an RGB Image from a URL and Resize:
python
Copy code
import requests
from PIL import Image
from io import BytesIO

url = "https://example.com/your_image.jpg"  # Replace with your image URL
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Resize the image to 224x224
img_resized = img.resize((224, 224))
Show a Grayscale Copy:
python
Copy code
import matplotlib.pyplot as plt

# Convert the image to grayscale
img_gray = img_resized.convert('L')

# Display grayscale image
plt.imshow(img_gray, cmap='gray')
plt.axis('off')
plt.show()
Convolve with 10 Random Filters and Show Filter Outputs:
python
Copy code
import numpy as np
from scipy.signal import convolve2d

# Generate 10 random 3x3 filters
filters = [np.random.rand(3, 3) for _ in range(10)]

# Convert the image to numpy array
img_array = np.array(img_resized)

# Apply each filter and display the results
for i, filter in enumerate(filters):
    filtered_image = convolve2d(img_array, filter, mode='valid')
    plt.subplot(2, 5, i+1)




    import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from scipy import signal
from scipy.signal import convolve2d
from skimage import data, color, io
import IPython

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from skimage.util import montage
from skimage.io import imread

import imageio as io

from skimage.transform import rescale, resize

def plot(x):
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap = 'gray')
    ax.axis('off')
    fig.set_size_inches(5, 5)
    plt.show()

### Original Image

image = io.imread("https://t4.ftcdn.net/jpg/02/40/33/95/360_F_240339505_8XvS2gEm3VcMN4nQIIBH2WqEtzPQUOsT.jpg")
image = image[:,:,:]

plot(image)

image.shape

### Resized Image

image_r = image[131:355, 252:476, :]



plot(image_r)

image_r.shape

#Averaging Method
image_g = np.mean(image_r, axis = 2)
plot(image_g)

#Channel Mixing Method
red_coefficient = 0.05
green_coefficient = 0.01
blue_coefficient = 0.01

image_c = (
    image_r[:,:, 0] * red_coefficient +
    image_r[:,:, 1] * green_coefficient +
    image_r[:,:, 2] * blue_coefficient
).astype(np.uint8)

plot(image_c)

#Red Channel Method
image_red = image_r[:,:,0]
plot(image_red)

#Luminance Method
image_l = np.dot(image_r[:, :, :3], [0.299, 0.587, 0.114]).astype(np.uint8)
plot(image_l)

### Convolve with 10 random filters

for i in range(10):

  filter_i = np.random.random((i+1,i+1))
  #filter_i = filters[i,:,:]
  image_i = signal.convolve2d(image_g, filter_i, mode='same')

  print(f"Filter: {i + 1}")
  plot(filter_i)
  print(f"Image: {i + 1}")
  plot(image_i)

### Filter Image by Hand

#### Mean Removal Filter

mean_removal = np.array([[-1, -1, -1],
                         [-1,  9, -1],
                         [-1, -1, -1]])


image_mean_removal = signal.convolve2d(image_g, mean_removal, mode='same')
plot(image_mean_removal)

#### Unsharp Masking Filter

unsharp_masking = np.array([[-1, -2, -1], [-2, 28, -2], [-1, -2, -1]])

image_unsharp_masking = signal.convolve2d(image_g, unsharp_masking, mode='same')
plot(image_unsharp_masking)

#### Gaussian Blur


gaussian_blur = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]]) / 16


image_gaussian_blur = signal.convolve2d(image_g, gaussian_blur, mode='same')
plot(image_gaussian_blur)

#### Emboss Filter

emboss = np.array([[-2, -1, 0],
                   [-1, 1, 1],
                   [0, 1, 2]])



image_emboss = signal.convolve2d(image_g, emboss, mode='same')
plot(image_emboss)

#### Sobel Filter




sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

image_sobel_x= signal.convolve2d(image_g, sobel_x, mode='same')
plot(image_sobel_x)
    plt.imshow(filtered_image, cmap='gray')
    plt.axis('off')
    plt.title(f'Filter {i+1}')
    
plt.show()
