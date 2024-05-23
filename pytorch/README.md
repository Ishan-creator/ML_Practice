# Nepali-Digit Prediction using PyTorch

## Introduction
In this project, I have created an Artificial Neural Network (ANN) model using PyTorch to predict Nepali digits. The dataset consists of images representing digits in the Nepali script.

## Data Format
The data is organized into two folders: Train and Test. Each folder contains images categorized into 10 classes, with each class containing 2000 images. The data format is similar to that of the MNIST dataset.

## Approach
1. **Custom Dataset Creation**: Firstly, I created a custom dataset where I loaded all the images and split them into training, validation, and testing sets. The training set comprises 70% of the data, the validation set 15%, and the testing set 15%. 

2. **Data Transformation**: I applied transformations to the images during dataset loading. These transformations included converting the images into tensors, resizing them to 32x32 pixels, and converting them to grayscale.

3. **Data Loader**: Utilizing PyTorch's DataLoader, I obtained batches of 64 images each to feed into the model during training.

4. **Model Architecture**: I experimented with different hyperparameters to tune the model's performance:
   
   | Hidden Layer Size | Weight Initialization | Optimizer | Learning Rate | Momentum | Activation Function |
   |-------------------|-----------------------|-----------|---------------|----------|---------------------|
   | 128               | Xavier_normal         | SGD       | 0.001         | 0.9      | ReLU                |
   | 256               | Xavier_uniform        | SGD       | 0.001         | 0.9      | ReLU                |
   | 512               | Normal                | SGD       | 0.001         | 0.9      | ReLU                |
   | 1024              | Zero                  | SGD       | 0.001         | 0.9      | ReLU                |

5. **Loss Function**: The loss function used for training the model was the Cross Entropy Loss, suitable for multi-class classification tasks.

6. **Scheduler for Early Stopping**: In addition to training the model, I implemented a scheduler for early stopping to prevent overfitting. If the validation accuracy does not increase for 10 consecutive epochs, the learning rate is reduced by a factor of 0.1. This reduction is applied for a maximum of 2 times before early stopping is triggered, terminating the training process.

This approach helps in optimizing the model's performance while preventing it from memorizing the training data excessively, leading to better generalization on unseen data.




## Evaluation


| Hidden Layer Size | Learning Rate | Batch Size | Epochs | Optimizer | Activation | Weight Initialization | Momentum | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss | Testing Accuracy |
|-------------------|---------------|------------|--------|------------|------------|-----------------------|----------|-------------------|---------------|---------------------|-----------------|------------------|
| 128               | 0.001         | 64         | 100    | SGD        | ReLU       | Xavier_normal         | 0.9      | 0.9997            | 0.0098        | 0.98                | 0.0615          | 97.32              |
| 256               | 0.001         | 64         | 100    | SGD        | ReLU       | Xavier_normal         | 0.9      | 1.0               | 0.0063        | 0.981               | 0.0752          | 95.34               |
| 512               | 0.001         | 64         | 100    | SGD        | ReLU       | Xavier_normal         | 0.9      | 1.0               | 0.0053        | 0.982               | 0.0578          | 94.62               |
| 1024              | 0.001         | 64         | 100    | SGD        | ReLU       | Xavier_normal         | 0.9      | 0.9999            | 0.0115        | 0.9793              | 0.0667          | 96.25               |
| 128               | 0.001         | 64         | 100    | SGD        | ReLU       | Normal                | 0.9      | 0.9957            | 0.1228        | 0.9313              | 4.7470          | 93.27            |
| 512               | 0.001         | 64         | 100    | SGD        | ReLU       | Normal                | 0.9      | 0.9996            | 0.0279        | 0.9393              | 12.8659         | 94.57            |
| 128               | 0.001         | 64         | 100    | SGD        | ReLU       | Zero                  | 0.9      | 0.9919            | 0.0442        | 0.9637              | 0.1308          | 96.67            |
| 256               | 0.001         | 64         | 100    | SGD        | ReLU       | Zero                  | 0.9      | 0.9933            | 0.0441        | 0.9667              | 0.1229          | 96.7             |


## Plots for visualization

![Alt Text](https://github.com/shailesh-olive/InternshipRepo/blob/9a3af61288e4f47ef198bd139de0313a3764f2b8/pytorch/images/SGD_RELU_1024_Xav_Nor_0.001_64.png) 

This is for the combination of SGD_RELU_1024_Xav_Nor_0.001_64

![Alt Text](https://github.com/shailesh-olive/InternshipRepo/blob/9a3af61288e4f47ef198bd139de0313a3764f2b8/pytorch/images/SGD_RELU_128_zeros_0.001_64.png) 

This is for the combintion of SGD_RELU_128_zeros_0.001_64

![Alt Text](https://github.com/shailesh-olive/InternshipRepo/blob/9a3af61288e4f47ef198bd139de0313a3764f2b8/pytorch/images/SGD_RELU_256_Nor_0.001_64.png) 

This is for the combination of SGD_RELU_256_Nor_0.001_64


From the above plots we can see that the performance of normal initialization is very poor tht can be verified from the  evaluation table as well . 




## Observation  and Conclusion
We can see that the training accuracy and validation accuracy is significantly high this indicate the chances of overfitting of the model. But the testing data is also 
above 95% indicating the model performance on unseen data as well. 
But when passing an image that is not the same format as the training and testing the model doesnt generalizes well. This is a big problem as our model isnt generalised for every kind of data.
This project demonstrates the application of PyTorch in building a simple ANN model for the prediction of Nepali digits. The provided code can be further extended or modified for more complex tasks or different datasets.


## Usage
1. **Installation**: Ensure you have PyTorch installed. If not, you can install it using `pip install torch`.

2. **Training**: Run the training script `train.py` to train the model. Make sure to specify the appropriate paths for the training and testing data.

3. **Evaluation**: Evaluate the trained model using the testing script `test.py`. Again, ensure correct paths are specified for the testing data.

