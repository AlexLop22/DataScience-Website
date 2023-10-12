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
    plt.imshow(filtered_image, cmap='gray')
    plt.axis('off')
    plt.title(f'Filter {i+1}')
    
plt.show()
