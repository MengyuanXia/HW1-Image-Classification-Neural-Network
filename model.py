import numpy as np
import gzip
import os
import struct
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        if self.activation == 'tanh':
            self.a1 = np.tanh(self.z1)
        elif self.activation == 'sigmoid':
            self.a1 = 1 / (1 + np.exp(-self.z1))
        else:
            raise ValueError("Unsupported activation function")
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, learning_rate, reg_strength):
        # Backward pass
        num_examples = X.shape[0]
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1
        delta3 /= num_examples

        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        if self.activation == 'tanh':
            delta2 = np.dot(delta3, self.W2.T) * (1 - np.power(self.a1, 2))
        elif self.activation == 'sigmoid':
            delta2 = np.dot(delta3, self.W2.T) * (self.a1 * (1 - self.a1))
        else:
            raise ValueError("Unsupported activation function")
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms
        dW2 += reg_strength * self.W2
        dW1 += reg_strength * self.W1

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, initial_learning_rate=0.01, reg_strength=0.01, num_epochs=40, batch_size=64, X_val=None, y_val=None, lr_decay=0.95):
        num_examples = X.shape[0]
        training_loss_history = []
        validation_accuracy_history = []
        best_val_accuracy = 0.0
        best_weights = None
        learning_rate = initial_learning_rate

        for epoch in range(num_epochs):
            # Shuffle training data for each epoch
            permutation = np.random.permutation(num_examples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            # Mini-batch SGD
            for i in range(0, num_examples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Ensure last batch size is correct
                if X_batch.shape[0] != batch_size:
                    batch_size = X_batch.shape[0]  # Update batch size for last batch

                # Forward pass
                probs = self.forward(X_batch)

                # Compute loss
                corect_logprobs = -np.log(probs[range(batch_size), y_batch])
                data_loss = np.sum(corect_logprobs)
                reg_loss = 0.5 * reg_strength * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
                loss = data_loss + reg_loss
                training_loss_history.append(loss)

                # Backward pass
                self.backward(X_batch, y_batch, learning_rate, reg_strength)

            # Compute validation accuracy if validation set is provided
            if X_val is not None and y_val is not None:
                y_val_pred = np.argmax(self.forward(X_val), axis=1)
                val_accuracy = np.mean(y_val_pred == y_val)
                validation_accuracy_history.append(val_accuracy)
                print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy}')

                # Save best weights
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_weights = (self.W1, self.b1, self.W2, self.b2)

            # Learning rate decay
            learning_rate *= lr_decay  # Decay learning rate

        if best_weights is not None:
            self.W1, self.b1, self.W2, self.b2 = best_weights  # Restore best weights

        return training_loss_history, validation_accuracy_history

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def save_weights(self, file_path):
        # Save model weights to a file using pickle
        with open(file_path, 'wb') as f:
            pickle.dump((self.W1, self.b1, self.W2, self.b2), f)

    @staticmethod
    def load_weights(file_path):
        # Load model weights from a file using pickle
        with open(file_path, 'rb') as f:
            W1, b1, W2, b2 = pickle.load(f)
        model = NeuralNetwork(W1.shape[0], W1.shape[1], W2.shape[1])  # Create a new model instance
        model.W1 = W1
        model.b1 = b1
        model.W2 = W2
        model.b2 = b2
        return model


def load_data():
    # Load Fashion-MNIST dataset
    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            _, num_data, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_data, rows * cols)
        return images

    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            _, num_data = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # Preprocess data
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Split train set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    print("Training samples:", X_train.shape[0])
    print("Validation samples:", X_val.shape[0])
    print("Training labels:", y_train.shape[0])
    print("Validation labels:", y_val.shape[0])

    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_images(images, labels, class_names):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.set_xlabel(class_names[labels[i]])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_learning_curve(training_loss_history, validation_accuracy_history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(training_loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(validation_accuracy_history)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()


def visualize_parameters(model):
    # Visualize model parameters (weights)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(model.W1, cmap='gray', aspect='auto')
    axes[0].set_title('Hidden Layer Weights')
    axes[1].imshow(model.W2, cmap='gray', aspect='auto')
    axes[1].set_title('Output Layer Weights')
    plt.show()

    # Visualize model parameters (weights and biases) using histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot histograms for hidden layer weights and biases
    axes[0, 0].hist(model.W1.flatten(), bins=50, color='b', alpha=0.7)
    axes[0, 0].set_title('Hidden Layer Weights Histogram')
    axes[0, 1].hist(model.b1.flatten(), bins=50, color='r', alpha=0.7)
    axes[0, 1].set_title('Hidden Layer Biases Histogram')

    # Plot histograms for output layer weights and biases
    axes[1, 0].hist(model.W2.flatten(), bins=50, color='g', alpha=0.7)
    axes[1, 0].set_title('Output Layer Weights Histogram')
    axes[1, 1].hist(model.b2.flatten(), bins=50, color='m', alpha=0.7)
    axes[1, 1].set_title('Output Layer Biases Histogram')

    plt.tight_layout()
    plt.show()


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Plot sample images
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']
    plot_images(X_train, y_train, class_names)

    # Hyperparameter search space
    learning_rates = [0.01, 0.1]
    hidden_sizes = [64, 128, 256]
    reg_strengths = [0.001, 0.01]

    # Record performance for different hyperparameters
    results = []

    for lr in learning_rates:
        for hs in hidden_sizes:
            for rs in reg_strengths:
                # Create and train model
                model = NeuralNetwork(input_size=28 * 28, hidden_size=hs, output_size=10, activation='sigmoid')
                training_loss_history, validation_accuracy_history = model.train(X_train, y_train, initial_learning_rate=lr,
                                                                                   reg_strength=rs, X_val=X_val,
                                                                                   y_val=y_val)

                # Test model performance
                test_accuracy = np.mean(model.predict(X_test) == y_test)

                # Record results
                results.append({
                    'learning_rate': lr,
                    'hidden_size': hs,
                    'reg_strength': rs,
                    'validation_accuracy': validation_accuracy_history[-1],
                    'test_accuracy': test_accuracy
                })
                print("Learning Rate:", lr, "Hidden Size:", hs,
                      "Regularization Strength:", rs, "Validation Accuracy:", validation_accuracy_history[-1],
                      "Test Accuracy:", test_accuracy)


    # Save weights of the best model
    best_hyperparameters = max(results, key=lambda x: x['validation_accuracy'])
    best_model = NeuralNetwork(input_size=28 * 28, hidden_size=best_hyperparameters['hidden_size'], output_size=10,
                               activation='sigmoid')
    best_model.train(X_train, y_train, initial_learning_rate=best_hyperparameters['learning_rate'],
                     reg_strength=best_hyperparameters['reg_strength'], X_val=X_val, y_val=y_val)
    best_model.save_weights('best_model_weights.pkl')

    # Output results
    for result in results:
        print("Learning Rate:", result['learning_rate'], "Hidden Size:", result['hidden_size'],
              "Regularization Strength:", result['reg_strength'], "Validation Accuracy:", result['validation_accuracy'],
              "Test Accuracy:", result['test_accuracy'])

    # Plot relevant analysis graphs
    plot_learning_curve(training_loss_history, validation_accuracy_history)
    visualize_parameters(model)


if __name__ == '__main__':
    main()
