package com.mlp;

import java.util.ArrayList;
import java.util.List;

import com.mlp.ActivationFunction.ActivationFunc;
import com.mlp.ActivationFunction.Linear;
import com.mlp.ActivationFunction.Sigmoid;
import com.mlp.LossFunction.BinaryCrossEntropy;
import com.mlp.LossFunction.LossFunction;
import com.mlp.LossFunction.MeanSquaredError;
import com.mlp.LossFunction.TaskType;
import com.mlp.Optimizer.Optimizer;

public class MLP {
    private List<Layer> layers;
    private Optimizer optimizer;
    private LossFunction lossFunction;
    private TaskType taskType;
    private WeighInit weighInitType;
    private ActivationFunc activationFunction;

    public MLP(Optimizer optimizer,
            TaskType taskType,
            LossFunction lossFunction,
            WeighInit weighType,
            ActivationFunc activationFunc,
            int... layerSizes) {
        this.optimizer = optimizer;
        this.activationFunction = activationFunc;
        this.weighInitType = weighType;
        this.taskType = taskType;
        this.layers = new ArrayList<>();

        ActivationFunc outputActivation;

        switch (taskType) {
            case REGRESSION:
                outputActivation = new Linear();
            case BINARY_CLASSIFICATION:
                outputActivation = new Sigmoid();
            default:
                throw new Error("Task type not supported");
        }

        for (int i = 0; i < layerSizes.length - 1; i++) {
            int numInputs = layerSizes[i];
            int numOutputs = layerSizes[i + 1];

            ActivationFunc currentActivation;
            if (i == layerSizes.length - 2) {
                currentActivation = outputActivation;
            } else {
                currentActivation = activationFunc;
            }

            layers.add(new Layer(numInputs, numOutputs, currentActivation, weighType));
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
        return this.lossFunction.compute(predicted, target);
    }

    public void backward(double[][] output) {
        Layer outputLayer = this.layers.get(this.layers.size() - 1);
        double[][] prediction = outputLayer.getActivatedData();

        double[][] deltaOutputMatrix;
        double[] predictionVector = prediction[0];
        double[] targetVector = output[0];
        double[] deltaVector = new double[predictionVector.length];

        for (int i = 0; i < predictionVector.length; i++) {
            deltaVector[i] = predictionVector[i] - targetVector[i];
        }
        deltaOutputMatrix = Matrix.rowVectorToMatrix(deltaVector);

        double[][] deltaForCurrentLayer = deltaOutputMatrix;
        double[][] weightsFromNextLayer = null;

        for (int i = this.layers.size() - 1; i >= 0; i--) {
            Layer currentLayer = this.layers.get(i);

            double[][] deltaToPropagate = currentLayer.backward(deltaForCurrentLayer, weightsFromNextLayer);

            weightsFromNextLayer = currentLayer.getWeights();
            deltaForCurrentLayer = deltaToPropagate;
        }
    }

    public void updateWeights() {
        if (this.layers == null)
            return;
        this.optimizer.update(this.layers);
    }

    public void train(double[][] trainingInputs, double[][] trainingTargets,
            double[][] validationInputs, double[][] validationTargets,
            int howManyEpochsMax,
            int patience,
            double stopIfLossBelow) {
        if (trainingInputs == null || trainingTargets == null || trainingInputs.length != trainingTargets.length) {
            System.out.println("ERROR: Training data or targets are bad!");
            return;
        }
        boolean checkValidation = (validationInputs != null && validationTargets != null);
        if (checkValidation && validationInputs.length != validationTargets.length) {
            System.out.println("ERROR: Validation data or targets are bad!");
            return;
        }
        if (howManyEpochsMax <= 0) {
            System.out.println("ERROR: Need to train for at least 1 epoch!");
            return;
        }
        if (patience > 0 && !checkValidation) {
            System.out.println("WARN: Patience needs validation data! Turning off early stopping.");
            patience = 0;
        }

        double bestScoreOnValidation = 999999999.0;
        int howManyEpochsNoImprovement = 0;
        String reasonWeStopped = "Finished all epochs (" + howManyEpochsMax + ")";
        boolean toldToStopEarly = false;

        System.out.println("=== Starting Training ===");
        System.out.println("Max Epochs: " + howManyEpochsMax);
        if (patience > 0) {
            System.out.println("Patience: " + patience);
        }
        if (stopIfLossBelow > 0) {
            System.out.println("Stop Loss Threshold: " + stopIfLossBelow);
        }
        System.out.println("-------------------------");

        for (int currentEpoch = 0; currentEpoch < howManyEpochsMax; currentEpoch++) {
            System.out.printf("Epoch %d/%d Start\n", currentEpoch + 1, howManyEpochsMax);

            double totalTrainLossThisEpoch = 0.0;

            for (int i = 0; i < trainingInputs.length; i++) {
                double[] currentInput = trainingInputs[i];
                double[] currentTarget = trainingTargets[i];

                double[][] inputMatrix = Matrix.rowVectorToMatrix(currentInput);
                double[][] targetMatrix = Matrix.rowVectorToMatrix(currentTarget);

                double[][] predictionOutput = this.forward(inputMatrix);

                double lossForThisExample = this.calculateLoss(predictionOutput, targetMatrix);
                totalTrainLossThisEpoch += lossForThisExample;

                this.backward(targetMatrix);

                this.updateWeights();
            }
            double averageTrainLoss = totalTrainLossThisEpoch / trainingInputs.length;
            System.out.printf("  Avg Train Loss: %.6f\n", averageTrainLoss);

            double averageValidationLoss = -1.0;
            if (checkValidation) {
                double totalValLossThisEpoch = 0.0;
                for (int i = 0; i < validationInputs.length; i++) {
                    double[][] valInputMatrix = Matrix.rowVectorToMatrix(validationInputs[i]);
                    double[][] valTargetMatrix = Matrix.rowVectorToMatrix(validationTargets[i]);

                    double[][] valPredictionOutput = this.forward(valInputMatrix);

                    totalValLossThisEpoch += this.calculateLoss(valPredictionOutput, valTargetMatrix);
                }
                averageValidationLoss = totalValLossThisEpoch / validationInputs.length;
                System.out.printf("  Avg Val Loss  : %.6f\n", averageValidationLoss);
            }


            double lossToUseForStopping = checkValidation ? averageValidationLoss : averageTrainLoss;
            if (stopIfLossBelow > 0 && lossToUseForStopping <= stopIfLossBelow) {
                reasonWeStopped = "Loss went below threshold (" + stopIfLossBelow + ") at epoch " + (currentEpoch + 1);
                toldToStopEarly = true;
                System.out.println("STOPPING: " + reasonWeStopped);
                break;
            }

            if (checkValidation && patience > 0) {
                if (averageValidationLoss < bestScoreOnValidation) {
                    System.out.println("  Validation loss improved!");
                    bestScoreOnValidation = averageValidationLoss;
                    howManyEpochsNoImprovement = 0;
                } else {
                    howManyEpochsNoImprovement++;
                    System.out.println(
                            "  Validation loss did not improve (" + howManyEpochsNoImprovement + "/" + patience + ")");
                    if (howManyEpochsNoImprovement >= patience) {
                        reasonWeStopped = "Stopped early because validation loss didn't improve for " + patience
                                + " epochs.";
                        toldToStopEarly = true;
                        System.out.println("STOPPING: " + reasonWeStopped);
                        break;
                    }
                }
            }
            System.out.println("---");

        }

        System.out.println("=========================");
        System.out.println("Training Finished!");
        System.out.println("Reason: " + reasonWeStopped);
        if (checkValidation && patience > 0) {
            System.out.printf("Best Validation Loss was: %.6f\n", bestScoreOnValidation);
        }
        System.out.println("=========================");

    }

}
