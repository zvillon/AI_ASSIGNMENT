# Project 3: MLP Implementation from Scratch

This project presents a from-scratch implementation of a Multi-Layer Perceptron (MLP) in Java. The MLP is one of the foundational architectures of deep learning, capable of solving both classification and regression problems.

## Architecture and Features

The project is designed to be modular and flexible, with the following components:

-   **`Layer`**: A class representing a layer of neurons, with its own weights, biases, and activation function.
-   **`MLP`**: The main class that assembles multiple layers to form the neural network.
-   **`ActivationFunction`**: An interface with several implementations (Sigmoid, ReLU, Softmax), allowing for easy changes to the model's non-linearity.
-   **`LossFunction`**: An interface for loss functions (Cross-Entropy for classification, MSE for regression).
-   **`Optimizer`**: A structure for optimization algorithms, with an implementation of Stochastic Gradient Descent (SGD) and Adam.

The model fully implements the forward pass and the backpropagation of error algorithm for learning.

## How to Run

The project is a standard Java application and uses Maven for dependency management and compilation.

1.  **Compilation:**
    ```bash
    mvn clean install
    ```
2.  **Execution:**
    *   The `Main` class contains an example of how to train and evaluate the MLP on a classification task.
    *   Run this class to see the model in action. The program loads data (e.g., from a CSV), trains the model, and displays the loss and accuracy over time.
    *   javac -d classes src/main/java/com/mlp/*.java src/main/java/com/mlp/Optimizer/*.java
    Inside of the first mlp directory then
    *   Then just java -cp classes com.mlp.Main

## Project Goal

The primary objective of this project was to master the mathematical and algorithmic concepts behind dense neural networks, including:
-   The matrix computations of the forward pass.
-   The chain rule for backpropagation.
-   The functioning of optimizers for weight updates.
-   The management of hyperparameters (number of layers, neurons, learning rate).

## Calculation result

```
=== Starting MNIST MLP Test (CSV Version) ===
Loading MNIST training data...
Loading CSV data from: mnist_train.csv (Limit: 10000, Header: true)
...read 10000 data rows
Finished reading 10000 data rows.
Loaded 10000 training samples.
Loading MNIST test data (for validation)...
Loading CSV data from: mnist_test.csv (Limit: 1666, Header: true)
Finished reading 1666 data rows.
Loaded 1666 validation samples.

Configuring MLP model...
Task: MULTICLASS_CLASSIFICATION, Output Activation: Softmax, Loss: CrossEntropyLoss
Adding Layer 0: 784 inputs, 128 outputs, Activation: ReLU, Init: HE_UNIFORM
Adding Layer 1: 128 inputs, 64 outputs, Activation: ReLU, Init: HE_UNIFORM
Adding Layer 2: 64 inputs, 10 outputs, Activation: Softmax, Init: HE_UNIFORM

Starting Training...
=== Starting Training ===
Task: MULTICLASS_CLASSIFICATION, Max Epochs: 10, Patience: 3, Stop Loss: 0.1000, Validation: true
Layers: 3, Optimizer: SGDOptimizer

---

Epoch 1/10 - Train Loss: 0.635400, Val Loss: 0.566743
Validation loss improved.

---

Epoch 2/10 - Train Loss: 0.474013, Val Loss: 0.532553
Validation loss improved.

---

Epoch 3/10 - Train Loss: 0.421357, Val Loss: 0.602376
Validation loss did not improve (1/3)

---

Epoch 4/10 - Train Loss: 0.376937, Val Loss: 0.492729
Validation loss improved.

---

Epoch 5/10 - Train Loss: 0.334236, Val Loss: 0.510769
Validation loss did not improve (1/3)

---

Epoch 6/10 - Train Loss: 0.343154, Val Loss: 0.587254
Validation loss did not improve (2/3)

---

Epoch 7/10 - Train Loss: 0.414607, Val Loss: 0.699817
Validation loss did not improve (3/3)
STOPPING: Validation loss did not improve for 3 epochs. Stopping at epoch 7.
=========================
Training Finished!
Reason: Validation loss did not improve for 3 epochs. Stopping at epoch 7.
Best Validation Loss achieved: 0.492729
=========================
Training completed in 44.45 seconds.

Evaluating final model on full test set...
(Reloading full test set for final evaluation)
Loading CSV data from: mnist_test.csv (Limit: -1, Header: true)
...read 10000 data rows
Finished reading 10000 data rows.
Evaluation Accuracy: 0.8854 (8854 / 10000 correct)
Evaluation completed in 2.66 seconds.

--- Testing prediction on individual samples ---
Sample Index: 0 | True Label: 7 | Predicted Label: 7 (Correct)
Sample Index: 42 | True Label: 4 | Predicted Label: 4 (Correct)
Sample Index: 101 | True Label: 0 | Predicted Label: 0 (Correct)
Sample Index: 500 | True Label: 3 | Predicted Label: 3 (Correct)
Sample Index: 4 | True Label: 4 | Predicted Label: 4 (Correct)

=== MNIST MLP Test (CSV Version) Completed ===
```