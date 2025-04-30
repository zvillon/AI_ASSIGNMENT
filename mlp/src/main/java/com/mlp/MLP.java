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

    public void train(double[][] inputs, double[][] targets, int epochs) {
        System.out.println("Starting training...");
        for (int e = 0; e < epochs; e++) {
            double epochLoss = 0;
            for (int i = 0; i < inputs.length; i++) {
                double[][] singleInput = Matrix.rowVectorToMatrix(inputs[i]);
                double[][] singleTarget = Matrix.rowVectorToMatrix(targets[i]);

                double[][] prediction = this.forward(singleInput);

                epochLoss += this.calculateLoss(prediction, singleTarget);

                this.backward(singleTarget);

                this.updateWeights();
            }
            if ((e + 1) % 100 == 0 || e == 0) {
                System.out.printf("Epoch %d/%d, Average Loss: %.6f%n",
                        e + 1, epochs, epochLoss / inputs.length);
            }
        }
        System.out.println("Training finished.");
    }

}
