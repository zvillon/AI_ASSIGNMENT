package com.mlp;

import java.util.ArrayList;
import java.util.List;

import com.mlp.ActivationFunction.*;
import com.mlp.LossFunction.*;
import com.mlp.Optimizer.Optimizer;

public class MLP {
    private List<Layer> layers;
    private Optimizer optimizer;
    private LossFunction lossFunction;
    private TaskType taskType;

    public MLP(Optimizer optimizer,
            TaskType taskType,
            WeighInit weighType,
            ActivationFunc hiddenActivationFunc,
            int... layerSizes) {

        if (layerSizes == null || layerSizes.length < 2) {
            throw new IllegalArgumentException("Need at least an input and output layer size.");
        }

        this.optimizer = optimizer;
        this.taskType = taskType;
        this.layers = new ArrayList<>();

        ActivationFunc outputActivation;
        switch (taskType) {
            case REGRESSION:
                outputActivation = new Linear();
                this.lossFunction = new MeanSquaredError();
                break;
            case BINARY_CLASSIFICATION:
                outputActivation = new Sigmoid();
                if (layerSizes[layerSizes.length - 1] != 1) {
                    System.out.println("Warning: Binary classification usually has 1 output neuron with Sigmoid. Found "
                            + layerSizes[layerSizes.length - 1]);
                }
                this.lossFunction = new CrossEntropyLoss();
                break;
            case MULTICLASS_CLASSIFICATION:
                outputActivation = new Softmax();
                this.lossFunction = new CrossEntropyLoss();
                break;
            default:
                throw new IllegalArgumentException("Task type not supported: " + taskType);
        }
        System.out.println("Task: " + taskType + ", Output Activation: " + outputActivation.getClass().getSimpleName()
                + ", Loss: " + lossFunction.getClass().getSimpleName());

        for (int i = 0; i < layerSizes.length - 1; i++) {
            int numInputs = layerSizes[i];
            int numOutputs = layerSizes[i + 1];

            ActivationFunc currentActivation;
            WeighInit currentInit = weighType;

            if (i == layerSizes.length - 2) {
                currentActivation = outputActivation;
            } else {
                currentActivation = hiddenActivationFunc;
            }

            System.out.printf("Adding Layer %d: %d inputs, %d outputs, Activation: %s, Init: %s\n",
                    i, numInputs, numOutputs, currentActivation.getClass().getSimpleName(), currentInit);
            layers.add(new Layer(numInputs, numOutputs, currentActivation, currentInit));
        }
    }

    public double[][] forward(double[][] networkInput) {
        double[][] currentData = networkInput;
        for (Layer curLayer : this.layers) {
            currentData = curLayer.forward(currentData);
        }
        return currentData;
    }

    public double calculateLoss(double[][] predicted, double[][] target) {
        if (this.lossFunction == null) {
            throw new IllegalStateException("Loss function has not been set.");
        }
        return this.lossFunction.compute(predicted, target);
    }

    public void backward(double[][] targetOutput) {
        if (this.layers == null || this.layers.isEmpty())
            return;

        Layer outputLayer = this.layers.get(this.layers.size() - 1);
        double[][] prediction = outputLayer.getActivatedData();

        if (prediction == null || targetOutput == null || prediction.length != targetOutput.length
                || prediction[0].length != targetOutput[0].length) {
            throw new IllegalArgumentException(
                    "Prediction and target dimensions mismatch during backward pass or prediction is null.");
        }

        double[][] deltaOutput = Matrix.substract(prediction, targetOutput);

        double[][] deltaForCurrentLayer = deltaOutput;
        double[][] weightsFromNextLayer = null;

        for (int i = this.layers.size() - 1; i >= 0; i--) {
            Layer currentLayer = this.layers.get(i);

            double[][] deltaToPropagate = currentLayer.backward(deltaForCurrentLayer, weightsFromNextLayer);

            weightsFromNextLayer = currentLayer.getWeights();
            deltaForCurrentLayer = deltaToPropagate;
        }
    }

    public void updateWeights() {
        if (this.optimizer == null) {
            System.err.println("Optimizer not set. Cannot update weights.");
            return;
        }
        this.optimizer.update(this.layers);
    }

    public void train(double[][] trainingInputs, double[][] trainingTargets,
            double[][] validationInputs, double[][] validationTargets,
            int maxEpochs,
            int patience,
            double stopLossThreshold) {
        if (trainingInputs == null || trainingTargets == null || trainingInputs.length != trainingTargets.length
                || trainingInputs.length == 0) {
            System.err.println("ERROR: Invalid training data or targets.");
            return;
        }
        boolean useValidation = (validationInputs != null && validationTargets != null
                && validationInputs.length == validationTargets.length && validationInputs.length > 0);
        if (patience > 0 && !useValidation) {
            System.out.println("WARN: Patience requires validation data. Disabling early stopping.");
            patience = 0;
        }
        if (maxEpochs <= 0) {
            System.err.println("ERROR: maxEpochs must be positive.");
            return;
        }

        System.out.printf(
                "=== Starting Training ===\nTask: %s, Max Epochs: %d, Patience: %d, Stop Loss: %.4f, Validation: %s\n",
                taskType, maxEpochs, patience, stopLossThreshold, useValidation);
        System.out.println("Layers: " + layers.size() + ", Optimizer: " + optimizer.getClass().getSimpleName());
        System.out.println("-------------------------");

        double bestValidationLoss = Double.POSITIVE_INFINITY;
        int epochsWithoutImprovement = 0;
        String stopReason = "Reached max epochs (" + maxEpochs + ")";

        for (int epoch = 0; epoch < maxEpochs; epoch++) {

            double epochTrainLoss = 0.0;

            for (int i = 0; i < trainingInputs.length; i++) {

                double[][] inputSample = Matrix.rowVectorToMatrix(trainingInputs[i]);
                double[][] targetSample = Matrix.rowVectorToMatrix(trainingTargets[i]);

                double[][] prediction = this.forward(inputSample);

                epochTrainLoss += this.calculateLoss(prediction, targetSample);

                this.backward(targetSample);

                this.updateWeights();
            }
            double avgTrainLoss = epochTrainLoss / trainingInputs.length;

            double avgValidationLoss = -1.0;
            if (useValidation) {
                double epochValidationLoss = 0.0;
                for (int i = 0; i < validationInputs.length; i++) {
                    double[][] valInputSample = Matrix.rowVectorToMatrix(validationInputs[i]);
                    double[][] valTargetSample = Matrix.rowVectorToMatrix(validationTargets[i]);
                    double[][] valPrediction = this.forward(valInputSample);
                    epochValidationLoss += this.calculateLoss(valPrediction, valTargetSample);
                }
                avgValidationLoss = epochValidationLoss / validationInputs.length;
                System.out.printf("Epoch %d/%d - Train Loss: %.6f, Val Loss: %.6f\n",
                        epoch + 1, maxEpochs, avgTrainLoss, avgValidationLoss);
            } else {
                System.out.printf("Epoch %d/%d - Train Loss: %.6f\n",
                        epoch + 1, maxEpochs, avgTrainLoss);
            }

            double lossToCheck = useValidation ? avgValidationLoss : avgTrainLoss;

            if (stopLossThreshold > 0 && lossToCheck <= stopLossThreshold) {
                stopReason = String.format("Loss (%.6f) reached threshold (%.4f) at epoch %d",
                        lossToCheck, stopLossThreshold, epoch + 1);
                System.out.println("STOPPING: " + stopReason);
                break;
            }

            if (useValidation && patience > 0) {
                if (avgValidationLoss < bestValidationLoss) {
                    System.out.println("  Validation loss improved.");
                    bestValidationLoss = avgValidationLoss;
                    epochsWithoutImprovement = 0;
                } else {
                    epochsWithoutImprovement++;
                    System.out.printf("  Validation loss did not improve (%d/%d)\n", epochsWithoutImprovement,
                            patience);
                    if (epochsWithoutImprovement >= patience) {
                        stopReason = String.format(
                                "Validation loss did not improve for %d epochs. Stopping at epoch %d.",
                                patience, epoch + 1);
                        System.out.println("STOPPING: " + stopReason);
                        break;
                    }
                }
            }
            System.out.println("---");
        }

        System.out.println("=========================");
        System.out.println("Training Finished!");
        System.out.println("Reason: " + stopReason);
        if (useValidation) {
            System.out.printf("Best Validation Loss achieved: %.6f\n", bestValidationLoss);
        }
        System.out.println("=========================");
    }
}