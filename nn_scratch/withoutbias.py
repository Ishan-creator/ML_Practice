
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd



class NeuralNetwork:
    def __init__(self, weight_initializer='he', activation_function='relu', input_num = 784,optimizer='sgd', learning_rate=0.01, hidden_units=128, output_units=10):
        self.weight_initializer = weight_initializer
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.input_num = input_num
        self.weights_input_hidden = self.initialize_weights(input_num, hidden_units)
        self.weights_hidden_output = self.initialize_weights(hidden_units, output_units)
        self.optimizer = optimizer

    def initialize_weights(self, input_num, output_num):
        if self.weight_initializer == 'he':
            standard_dev = np.sqrt(2 / input_num)
            return np.random.normal(0, standard_dev, (input_num, output_num))
        elif self.weight_initializer == 'xavier':
            standard_dev = np.sqrt(2 / (input_num + output_num))
            return np.random.normal(0, standard_dev, (input_num, output_num))
        elif self.weight_initializer == 'zero':
            return np.zeros((input_num, output_num))
        elif self.weight_initializer == 'random':
            return np.random.randn(input_num, output_num)
        else:
            raise ValueError("Invalid weight initialization method")


    def apply_activation(self, x):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'gelu':
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        elif self.activation_function == 'leaky_relu':
            alpha = 0.01
            return np.where(x > 0, x, alpha * x)
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Invalid activation function")

    def activation_derivative(self, x):
        if self.activation_function == 'sigmoid':
            return x * (1 - x)
        elif self.activation_function == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_function == 'gelu':
            return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))) 
                          + (np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))) * (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))) ** 2))
        elif self.activation_function == 'leaky_relu':
            alpha = 0.01
            return np.where(x > 0, 1, alpha)
        elif self.activation_function == 'tanh':
            return 1 - x ** 2
        else:
            raise ValueError("Invalid activation function")

    def initialize_optimizer(self):
        if self.optimizer == 'sgd':
            pass

        elif self.optimizer == 'adam':

            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.m_input_hidden = np.zeros_like(self.weights_input_hidden)
            self.v_input_hidden = np.zeros_like(self.weights_input_hidden)
            self.m_hidden_output = np.zeros_like(self.weights_hidden_output)
            self.v_hidden_output = np.zeros_like(self.weights_hidden_output)

        elif self.optimizer == 'rmsprop':
            self.beta = 0.9
            self.epsilon = 1e-8
            self.rms_input_hidden = np.zeros_like(self.weights_input_hidden)
            self.rms_hidden_output = np.zeros_like(self.weights_hidden_output)

        else:
            raise ValueError('Invalid optimizer choice')

    def forward_propagation(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden)
        self.hidden_layer_output = self.apply_activation(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.output_layer_output = self.apply_activation(self.output_layer_input)
        return self.output_layer_output

    def backward_propagation(self, X, y):
        d_loss_output = 2 * (self.output_layer_output - y) / y.shape[0]
        d_output_activation = self.activation_derivative(self.output_layer_output)
        d_output_weights = np.dot(self.hidden_layer_output.T, d_loss_output * d_output_activation)
        d_hidden_activation = self.activation_derivative(self.hidden_layer_output)
        d_hidden_weights = np.dot(X.T, np.dot(d_loss_output * d_output_activation, self.weights_hidden_output.T) * d_hidden_activation)      
        return d_hidden_weights, d_output_weights

    def update_weights(self, d_hidden_weights, d_output_weights):
        if self.optimizer == 'sgd':       
            self.weights_input_hidden -= self.learning_rate * d_hidden_weights
            self.weights_hidden_output -= self.learning_rate * d_output_weights

        elif self.optimizer == 'adam':
            self.m_input_hidden = self.beta1 * self.m_input_hidden + (1 - self.beta1) * d_hidden_weights
            self.v_input_hidden = self.beta2 * self.v_input_hidden + (1 - self.beta2) * (d_hidden_weights ** 2)
            m_hat_input_hidden = self.m_input_hidden / (1 - self.beta1)
            v_hat_input_hidden = self.v_input_hidden / (1 - self.beta2)
            self.weights_input_hidden -= self.learning_rate * m_hat_input_hidden / (np.sqrt(v_hat_input_hidden) + self.epsilon)

            self.m_hidden_output = self.beta1 * self.m_hidden_output + (1 - self.beta1) * d_output_weights
            self.v_hidden_output = self.beta2 * self.v_hidden_output + (1 - self.beta2) * (d_output_weights ** 2)
            m_hat_hidden_output = self.m_hidden_output / (1 - self.beta1)
            v_hat_hidden_output = self.v_hidden_output / (1 - self.beta2)
            self.weights_hidden_output -= self.learning_rate * m_hat_hidden_output / (np.sqrt(v_hat_hidden_output) + self.epsilon)  

        elif self.optimizer == 'rmsprop':
            self.rms_input_hidden = self.beta * self.rms_input_hidden + (1 - self.beta) * (d_hidden_weights ** 2)
            self.rms_hidden_output = self.beta * self.rms_hidden_output + (1 - self.beta) * (d_output_weights ** 2)
            self.weights_input_hidden -= self.learning_rate * d_hidden_weights / (np.sqrt(self.rms_input_hidden) + self.epsilon)
            self.weights_hidden_output -= self.learning_rate * d_output_weights / (np.sqrt(self.rms_hidden_output) + self.epsilon)

            
        
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, patience=3):


        self.initialize_optimizer()
        train_losses = []
        val_losses = []
        train_accuracies = []  # Add this line
        val_accuracies = []
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            total_loss = 0.0
            correct_train = 0  # Add this line
            for i in range(0, X_train.shape[0], batch_size):
                y_pred = self.forward_propagation(X_train[i:i+batch_size])
                loss = np.mean((y_pred - y_train[i:i+batch_size]) ** 2)
                total_loss += loss
                d_hidden_weights, d_output_weights = self.backward_propagation(X_train[i:i+batch_size], y_train[i:i+batch_size])
                self.update_weights(d_hidden_weights, d_output_weights)
                
                # Compute training accuracy
                train_pred_label = np.argmax(y_pred, axis=1)
                true_train_label = np.argmax(y_train[i:i+batch_size], axis=1)
                correct_train += np.sum(train_pred_label == true_train_label)  # Count correct predictions

            train_accuracy = correct_train / X_train.shape[0]  # Compute accuracy for the epoch
            train_accuracies.append(train_accuracy)  # Append to the list

            val_pred = self.forward_propagation(X_val)
            val_loss = np.mean((val_pred - y_val) ** 2)
            val_losses.append(val_loss)
            val_pred_label = np.argmax(val_pred, axis=1)
            true_val_label = np.argmax(y_val, axis=1)
            val_accuracy = np.mean(val_pred_label == true_val_label)
            val_accuracies.append(val_accuracy)

            train_losses.append(total_loss / (X_train.shape[0] / batch_size))
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Training Accuracy: {train_accuracy}")

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     epochs_no_improve = 0
            # else:
            #     epochs_no_improve += 1
            #     if epochs_no_improve == patience:
            #         print(f"Early stopping at epoch {epoch+1}")
            #         break

        return train_losses, val_losses, train_accuracies, val_accuracies  # Modify the return statement



# Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

val_split = 0.1
val_size = int(len(X_train) * val_split)
X_val = X_train[:val_size]
y_val = y_train[:val_size]
X_train = X_train[val_size:]
y_train = y_train[val_size:]

nn = NeuralNetwork(weight_initializer='random', activation_function='tanh', learning_rate=0.001, hidden_units=128, output_units=10, optimizer='adam')
train_losses, val_losses, val_accuracies, train_accuracies = nn.train(X_train, y_train, X_val, y_val, epochs=1000, batch_size=64)





plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()


plt.subplot(1, 2, 2)

plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Training Accuracy')
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('training and validation Accuracy')
plt.legend()

plt.tight_layout() 
plt.savefig('random_sigmoid_adam_without_bias for 0.001.png') 
plt.close()

test_predictions = nn.forward_propagation(X_test)
test_predictions_labels = np.argmax(test_predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(true_labels, test_predictions_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('CM_random_sigmoid_adam_without_bias for 0.001.png') 
plt.close()