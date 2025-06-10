[19:32:20] ➜  AI_ASSIGNMENT git:(main) ✗ java -cp mlp/src/main/java/classes com.mlp.Main 
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
-------------------------
Epoch 1/10 - Train Loss: 0.663055, Val Loss: 0.534841
  Validation loss improved.
---
Epoch 2/10 - Train Loss: 0.443984, Val Loss: 0.493896
  Validation loss improved.
---
Epoch 3/10 - Train Loss: 0.357731, Val Loss: 0.445581
  Validation loss improved.
---
Epoch 4/10 - Train Loss: 0.322992, Val Loss: 0.487177
  Validation loss did not improve (1/3)
---
Epoch 5/10 - Train Loss: 0.277817, Val Loss: 0.669097
  Validation loss did not improve (2/3)
---
Epoch 6/10 - Train Loss: 0.244368, Val Loss: 0.450196
  Validation loss did not improve (3/3)
STOPPING: Validation loss did not improve for 3 epochs. Stopping at epoch 6.
=========================
Training Finished!
Reason: Validation loss did not improve for 3 epochs. Stopping at epoch 6.
Best Validation Loss achieved: 0.445581
=========================
Training completed in 36.54 seconds.

Evaluating final model on full test set...
(Reloading full test set for final evaluation)
  Loading CSV data from: mnist_test.csv (Limit: -1, Header: true)
    ...read 10000 data rows
  Finished reading 10000 data rows.
Evaluation Accuracy: 0.9149 (9149 / 10000 correct)
Evaluation completed in 2.62 seconds.

--- Testing prediction on individual samples ---
Sample Index: 0 | True Label: 7 | Predicted Label: 7 (Correct)
Sample Index: 42 | True Label: 4 | Predicted Label: 4 (Correct)
Sample Index: 101 | True Label: 0 | Predicted Label: 0 (Correct)
Sample Index: 500 | True Label: 3 | Predicted Label: 3 (Correct)
Sample Index: 4 | True Label: 4 | Predicted Label: 4 (Correct)

=== MNIST MLP Test (CSV Version) Completed ===