import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import requests

url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Claude_Monet%2C_Saint-Georges_majeur_au_cr%C3%A9puscule.jpg/800px-Claude_Monet%2C_Saint-Georges_majeur_au_cr%C3%A9puscule.jpg'
im = imread(url)
plt.imshow(im);
im = resize(im,(512,512))
plt.imshow(im);
im.shape
plt.imshow(im[:,:,0],cmap='gray')
