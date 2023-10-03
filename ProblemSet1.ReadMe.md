#load MINIST


#Part1

%%capture
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from skimage.util import montage
!pip install wandb
import wandb as wb
from skimage.io import imread



def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))

def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))

def plot(x):
    if type(x) == torch.Tensor :
        x = x.cpu().detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap = 'gray')
    ax.axis('off')
    fig.set_size_inches(7, 7)
    plt.show()

def montage_plot(x):
    x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    plot(montage(x))

# #MNIST
train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

#KMNIST
#train_set = datasets.KMNIST('./data', train=True, download=True)
#test_set = datasets.KMNIST('./data', train=False, download=True)

#Fashion MNIST
# train_set = datasets.FashionMNIST('./data', train=True, download=True)
# test_set = datasets.FashionMNIST('./data', train=False, download=True)



X = train_set.data.numpy()
X_test = test_set.data.numpy()
Y = train_set.targets.numpy()
Y_test = test_set.targets.numpy()

X = X[:,None,:,:]/255
X_test = X_test[:,None,:,:]/255

X.shape

x = X[3,0,:,:]

plt.imshow(x)

plot(x)

x.shape

x.shape[0]

x.shape[1]

x = x.reshape(x.shape[0]*x.shape[1],1)

x.shape

x = x.reshape(28,28)

plot(x)

plot(X[100,0,:,:])

Y[120]

X[0:25,0,:,:].shape

montage_plot(X[125:150,0,:,:])

#Run random y=mx model on MNIST




import numpy as np
import matplotlib.pyplot as plt

# Generate a random slope (m) between -10 and 10
m = np.random.uniform(-10, 10)

# Generate random x values
x = np.random.uniform(-20, 20, 100)

# Calculate y = mx
y = m * x

# Plot the line y = mx
plt.plot(x, y, label=f'y = {m:.2f}x')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Random y = mx Model')
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a random slope (m) and intercept (b)
m = np.random.uniform(-10, 10)
b = np.random.uniform(-20, 20)

# Generate random x values
x = np.random.uniform(-20, 20, 100)

# Calculate corresponding y values
y = m * x + b

# Generate labels based on whether points are above or below the line
labels = (y > np.median(y)).astype(int)  # 1 if above, 0 if below

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x.reshape(-1, 1), labels, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the labels
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plot the data points and the decision boundary
plt.scatter(x, labels, color='blue', label='Actual labels')
plt.plot(x, model.predict(x.reshape(-1, 1)), color='red', label='Predicted labels')
plt.xlabel('x')
plt.ylabel('Label')
plt.legend()
plt.show()
