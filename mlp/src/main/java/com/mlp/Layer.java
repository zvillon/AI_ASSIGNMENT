package com.mlp;

import java.util.Arrays;
import java.util.Random;

import com.mlp.ActivationFunction.ActivationFunc;

public class Layer {

    int numInputs;
    int numOutputs;
    double[][] weights;
    double[] biases;

    ActivationFunc activationFunction;

    double[][] lastInput;
    double[][] lastZ;
    double[][] lastActivation;

    double[][] weightGradients;
    double[] biasGradients;
    double[][] delta;

    private static final Random rand = new Random(System.currentTimeMillis());

    public Layer(int numInputs, int numOutputs, ActivationFunc activation) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.activationFunction = activation;

        this.weights = new double[numOutputs][numInputs];
        this.biases = new double[numOutputs];
        this.weightGradients = new double[numOutputs][numInputs];
        this.biasGradients = new double[numOutputs];

        double limit = Math.sqrt(6.0 / (numInputs + numOutputs));
        for (int i = 0; i < numOutputs; i++) {
            for (int j = 0; j < numInputs; j++) {
                weights[i][j] = (rand.nextDouble() * 2.0 - 1.0) * limit;
            }
            biases[i] = 0.0;
        }
    }

    public double[][] forward(double[][] inputs) {
        if (inputs == null || inputs.length != 1 || inputs[0].length != numInputs) {
            throw new IllegalArgumentException("Input matrix dimensions incorrect. Expected [1][" + numInputs + "]");
        }
        this.lastInput = inputs;

        double[][] weightsTransposed = Matrix.transpose(this.weights);

        double[][] z_intermediate = Matrix.multiply(this.lastInput, weightsTransposed);

        this.lastZ = Matrix.addBiasVectorToRows(z_intermediate, this.biases);
        this.lastActivation = Matrix.applyFunc(this.lastZ, activationFunction::activate);

        return this.lastActivation;
    }

    public double[][] backward(double[][] deltaFromNextLayer, double[][] weightsFromNextLayer) {
        double[][] errorSignalPropagated;

        if (weightsFromNextLayer == null) {
            this.delta = deltaFromNextLayer;
        } else {
            errorSignalPropagated = Matrix.multiply(deltaFromNextLayer, weightsFromNextLayer);

            double[][] activationDerivative = Matrix.applyFunc(this.lastActivation, activationFunction::derivative);

            this.delta = Matrix.multiplyElementWise(errorSignalPropagated, activationDerivative);
        }
        double[][] deltaTransposed = Matrix.transpose(this.delta);
        this.weightGradients = Matrix.multiply(deltaTransposed, this.lastInput);

        if (this.delta != null && this.delta.length > 0) {
            this.biasGradients = Arrays.copyOf(this.delta[0], this.delta[0].length);
        }


        double[][] deltaForPreviousLayer = Matrix.multiply(this.delta, this.weights);

        return deltaForPreviousLayer;
    }

    public void updateWeights(double learningRate) {
        double[][] scaledWeightGradients = Matrix.multiply(this.weightGradients, learningRate);
        this.weights = Matrix.substract(this.weights, scaledWeightGradients);

        for (int i = 0; i < numOutputs; i++) {
            this.biases[i] -= learningRate * this.biasGradients[i];
        }
    }
}