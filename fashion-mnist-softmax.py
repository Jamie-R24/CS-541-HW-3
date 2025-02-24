import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class SoftmaxClassifier:
    def __init__(self, input_size: int, num_classes: int, learning_rate: float, alpha: float, batch_size: int):
        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Using small random values for better convergence
        self.W = np.random.randn(input_size, num_classes) * 0.01
        self.b = np.zeros(num_classes)
    
    def softmax(self, z: np.ndarray) -> np.ndarray:
        """Compute softmax values for each set of scores in z."""
    
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass to compute class probabilities."""
        
        z = np.dot(X, self.W) + self.b
        
        return self.softmax(z)
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute cross-entropy loss with L2 regularization."""
        N = X.shape[0]
        probs = self.forward(X)
        
        
        y_one_hot = np.zeros((N, self.num_classes))
        y_one_hot[np.arange(N), y] = 1
        
        
        correct_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(correct_logprobs) / N
        
        
        reg_loss = 0.5 * self.alpha * np.sum(self.W * self.W)
        
        return data_loss + reg_loss, probs
    
    def backward(self, X: np.ndarray, y: np.ndarray, probs: np.ndarray) -> None:
        """Compute gradients and update parameters."""
        N = X.shape[0]
        
        dscores = probs.copy()
        dscores[range(N), y] -= 1
        dscores /= N
       
        dW = np.dot(X.T, dscores)
        db = np.sum(dscores, axis=0)
        
        dW += self.alpha * self.W

        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, 
              num_epochs: int, verbose: bool = True) -> List[float]:
        
        """Train the model using mini-batch SGD."""
        num_train = X_train.shape[0]
        train_losses = []
        best_val_acc = 0
        best_params = None
        
        for epoch in range(num_epochs):
            
            indices = np.random.permutation(num_train)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0, num_train, self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                probs = self.forward(X_batch)
                loss, _ = self.compute_loss(X_batch, y_batch)
                train_losses.append(loss)
                self.backward(X_batch, y_batch, probs)

            val_loss, val_acc = self.evaluate(X_val, y_val)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = (self.W.copy(), self.b.copy())
            
            if verbose and epoch % 5 == 0:
                print(f"Epoch {epoch}: val_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}")

        if best_params is not None:
            self.W, self.b = best_params
        
        return train_losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Compute loss and accuracy."""
        loss, probs = self.compute_loss(X, y)
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y)
        return loss, accuracy

def train_and_evaluate():

    train_images = np.load('fashion_mnist_train_images.npy')
    train_labels = np.load('fashion_mnist_train_labels.npy')
    test_images = np.load('fashion_mnist_test_images.npy')
    test_labels = np.load('fashion_mnist_test_labels.npy')
    
    #Putting all the results in a text file for easier access and readability
    with open('results.txt', 'w') as f:
        f.write("Fashion MNIST Classification Results\n")
        f.write("===================================\n\n")
        
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        X_train, X_val, y_train, y_val = train_test_split(
            train_images, train_labels, test_size=0.2, random_state=42
        )

        learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
        alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        batch_sizes = [32, 64, 128, 256]
        
        best_val_acc = 0
        best_params = None
        best_model = None
        
        f.write("Hyperparameter Search Results:\n")
        f.write("-----------------------------\n")

        for lr in learning_rates:
            for alpha in alphas:
                for batch_size in batch_sizes:
                    result = f"\nTrying lr={lr}, alpha={alpha}, batch_size={batch_size}"
                    print(result)
                    f.write(result + '\n')
                    
                    model = SoftmaxClassifier(
                        input_size=784,
                        num_classes=10,
                        learning_rate=lr,
                        alpha=alpha,
                        batch_size=batch_size
                    )
  
                    model.train(X_train, y_train, X_val, y_val, num_epochs=5, verbose=False)

                    val_loss, val_acc = model.evaluate(X_val, y_val)
                    result = f"Validation accuracy: {val_acc:.4f}"
                    print(result)
                    f.write(result + '\n')
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_params = {
                            'learning_rate': lr,
                            'alpha': alpha,
                            'batch_size': batch_size
                        }
                        best_model = model
        
        result = "\nBest hyperparameters found:"
        print(result)
        f.write('\n' + result + '\n')
        for param, value in best_params.items():
            result = f"{param}: {value}"
            print(result)
            f.write(result + '\n')

        final_model = SoftmaxClassifier(
            input_size=784,
            num_classes=10,
            **best_params
        )

        train_losses = final_model.train(X_train, y_train, X_val, y_val, num_epochs=30)

        test_loss, test_acc = final_model.evaluate(test_images, test_labels)
        
        f.write('\nFinal Test Set Performance:\n')
        f.write('-------------------------\n')
        f.write(f"Cross-entropy loss: {test_loss:.4f}\n")
        f.write(f"Classification accuracy: {test_acc:.4f}\n")
        
        print(f"\nTest set performance:")
        print(f"Cross-entropy loss: {test_loss:.4f}")
        print(f"Classification accuracy: {test_acc:.4f}")
        
        f.write("\nResults have been saved to results.txt")
    
    return test_loss, test_acc, train_losses

if __name__ == "__main__":
    test_loss, test_acc, train_losses = train_and_evaluate()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')  # Save the plot as an image
    plt.show()
