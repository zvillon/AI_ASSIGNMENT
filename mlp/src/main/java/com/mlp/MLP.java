package com.mlp;

import java.util.List;

import com.mlp.ActivationFunction.ActivationFunc;
import com.mlp.LossFunction.BinaryCrossEntropy;
import com.mlp.LossFunction.LossFunction;
import com.mlp.LossFunction.MeanSquaredError;
import com.mlp.LossFunction.TaskType;

public class MLP {
    private List<Layer> network;
    private double learningRate;
    private LossFunction lossFunction;
    private ActivationFunc activationFunction;
    private WeighInit weighInitType;

    public MLP(int layerSize, double learningRate, ActivationFunc activationFunc, TaskType type, WeighInit weighInit) {
        this.learningRate = learningRate;
        this.activationFunction = activationFunc;
        this.weighInitType = weighInit;

        switch (type) {
            case BINARY_CLASSIFICATION:
                this.lossFunction = new BinaryCrossEntropy();
                break;
            case REGRESSION:
                this.lossFunction = new MeanSquaredError();
                break;
            default:
                throw new ExceptionInInitializerError("Didn't recognize the TaskType");
        }

    }

    public double[][] forward(double[][] networkInput) {
        double[][] currentData = networkInput;
        for (Layer curLayer : this.network) {
            currentData = curLayer.forward(currentData);
        }
        return currentData;
    }

    public double calculateLoss(double[][] predicted, double[][] target) {
        return this.lossFunction.compute(predicted, target);
    }

    public void backward(double[][] output) {
        Layer outputLayer = this.network.get(this.network.size() - 1);
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

        for (int i = this.network.size() - 1; i >= 0; i--) {
            Layer currentLayer = this.network.get(i);

            double[][] deltaToPropagate = currentLayer.backward(deltaForCurrentLayer, weightsFromNextLayer);

            weightsFromNextLayer = currentLayer.getWeights();
            deltaForCurrentLayer = deltaToPropagate;
        }
    }

    public void updateWeights() {
        if (this.network == null)
            return;
        for (Layer layer : this.network) {
            layer.updateWeights(this.learningRate);
        }
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
