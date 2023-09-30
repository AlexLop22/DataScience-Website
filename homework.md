Load MNIST and show a montage of the images.
Run a random y=mx model on MNIST
Train random walk model to at least 75% accuracy 
Link below

https://colab.research.google.com/drive/1IRnqlgQPu5-vNSR5rB4xPzKHRAVcNBt1#scrollTo=f0EfR2jZP7wy 

# Random y=mx Model Report

## Introduction

This report presents a random y=mx model, where 'y' is dependent on 'x' with a randomly generated slope 'm'. The model is a simple linear relationship defined by the equation: y = mx.

## Model Description

The model follows the linear equation: y = mx, where:
- 'y' is the dependent variable,
- 'x' is the independent variable, and
- 'm' is a random slope.

## Data Generation

Random 'x' values were generated within the range [-20, 20]. The corresponding 'y' values were calculated using the formula: y = mx + b, where 'm' is the random slope and 'b' is a random intercept.

## Data Visualization

The data points and the random line y = mx were plotted to visualize the model.

![Data Visualization](path_to_visualization_image)

## Model Training

A logistic regression model was trained to classify points based on whether they are above or below the random line y = mx.

## Model Evaluation

The accuracy of the model was evaluated using the accuracy score, which indicated how well the model classified the points.

Accuracy: 99.00%

## Conclusion

Although the model was able to classify points based on the random y=mx relationship, it's important to note that achieving a specific accuracy in this scenario is not possible due to the inherently random nature of the model.

