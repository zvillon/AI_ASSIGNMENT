package com.mlp;

import java.util.Arrays;
import java.util.Random;

import com.mlp.ActivationFunction.ActivationFunc;
import com.mlp.ActivationFunction.Softmax;

public class Layer {

    private int numInputs;
    private int numOutputs;
    private double[][] weights;
    private double[] biases;

    private ActivationFunc activationFunction;

    private double[][] lastInput;
    private double[][] weightedSum;
    private double[][] activatedData;

    private double[][] weightGradients;
    private double[] biasGradients;
    private double[][] delta;

    private static final Random rand = new Random(System.currentTimeMillis());

    public Layer(int numInputs, int numOutputs, ActivationFunc activation, WeighInit initMethod) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.activationFunction = activation;

        this.weights = new double[numOutputs][numInputs];
        this.biases = new double[numOutputs];

        initializeWeightsAndBiases(initMethod);

        this.weightGradients = new double[numOutputs][numInputs];
        this.biasGradients = new double[numOutputs];
    }

    private void initializeWeightsAndBiases(WeighInit initMethod) {
        switch (initMethod) {
            case GLOROT_UNIFORM:
                double limitGlorot = Math.sqrt(6.0 / (numInputs + numOutputs));
                for (int i = 0; i < numOutputs; i++) {
                    for (int j = 0; j < numInputs; j++) {
                        weights[i][j] = (rand.nextDouble() * 2.0 - 1.0) * limitGlorot;
                    }
                    biases[i] = 0.0;
                }
                break;
            case HE_UNIFORM:
                double limitHe = Math.sqrt(6.0 / numInputs);
                for (int i = 0; i < numOutputs; i++) {
                    for (int j = 0; j < numInputs; j++) {
                        weights[i][j] = (rand.nextDouble() * 2.0 - 1.0) * limitHe;
                    }
                    biases[i] = 0.0;
                }
                break;
            case RANDOM_NORMAL:
                double stdDev = 0.01;
                for (int i = 0; i < numOutputs; i++) {
                    for (int j = 0; j < numInputs; j++) {
                        weights[i][j] = rand.nextGaussian() * stdDev;
                    }
                    biases[i] = 0.0;
                }
                break;
            case RANDOM_UNIFORM:
                double range = 0.01;
                for (int i = 0; i < numOutputs; i++) {
                    for (int j = 0; j < numInputs; j++) {
                        weights[i][j] = (rand.nextDouble() * 2.0 - 1.0) * range;
                    }
                    biases[i] = 0.0;
                }
                break;
            case ZEROS:
            default:
                for (int i = 0; i < numOutputs; i++) {
                    for (int j = 0; j < numInputs; j++) {
                        weights[i][j] = 0.0;
                    }
                    biases[i] = 0.0;
                }
                break;
        }
    }

    public double[][] forward(double[][] inputs) {
        if (inputs == null || inputs.length == 0 || inputs[0].length != numInputs) {
            throw new IllegalArgumentException("Input matrix dimensions incorrect. Expected [batch_size][" + numInputs
                    + "], got [" + (inputs == null ? 0 : inputs.length) + "]["
                    + (inputs == null || inputs.length == 0 ? 0 : inputs[0].length) + "]");
        }
        this.lastInput = inputs;

        double[][] weightsTransposed = Matrix.transpose(this.weights);
        this.weightedSum = Matrix.multiply(this.lastInput, weightsTransposed);

        this.weightedSum = Matrix.addBiasVectorToRows(this.weightedSum, this.biases);

        if (this.activationFunction instanceof Softmax) {
            this.activatedData = Softmax.activateMatrix(this.weightedSum);
        } else {
            this.activatedData = Matrix.applyFunc(this.weightedSum, this.activationFunction::activate);
        }

        return this.activatedData;
    }

    public double[][] backward(double[][] deltaOrPropagatedError, double[][] weightsFromNextLayer) {

        if (weightsFromNextLayer == null) {
            this.delta = deltaOrPropagatedError;
        } else {
            double[][] activationDerivative;
            if (this.activationFunction instanceof Softmax) {
                activationDerivative = Matrix.applyFunc(this.activatedData, activationFunction::derivative);
            } else {
                activationDerivative = Matrix.applyFunc(this.activatedData, activationFunction::derivative);
            }
            if (deltaOrPropagatedError.length != activationDerivative.length
                    || deltaOrPropagatedError[0].length != activationDerivative[0].length) {
                throw new IllegalStateException(String.format(
                        "Dimension mismatch in Layer.backward: propagatedError [%d,%d] vs activationDerivative [%d,%d]",
                        deltaOrPropagatedError.length, deltaOrPropagatedError[0].length,
                        activationDerivative.length, activationDerivative[0].length));
            }

            this.delta = Matrix.multiplyElementWise(deltaOrPropagatedError, activationDerivative);
        }
        if (this.delta == null || this.lastInput == null) {
            throw new IllegalStateException(
                    "Delta or lastInput is null during gradient calculation in Layer.backward.");
        }

        double[][] deltaTransposed = Matrix.transpose(this.delta);
        this.weightGradients = Matrix.multiply(deltaTransposed, this.lastInput);

        this.biasGradients = Matrix.sumColumns(this.delta);

        int batchSize = this.lastInput.length;
        if (batchSize > 0) {
            this.weightGradients = Matrix.multiply(this.weightGradients, 1.0 / batchSize);
            for (int i = 0; i < this.biasGradients.length; ++i) {
                this.biasGradients[i] /= batchSize;
            }
        }

        double[][] deltaForPreviousLayer = Matrix.multiply(this.delta, this.weights);

        return deltaForPreviousLayer;
    }

    public double[][] getActivatedData() {
        return this.activatedData;
    }

    public double[][] getWeights() {
        return this.weights;
    }

    public double[] getBiases() {
        return this.biases;
    }

    public double[][] getWeightsGradient() {
        return this.weightGradients;
    }

    public double[] getBiasGradient() {
        return this.biasGradients;
    }

    public void setWeights(double[][] weights) {
        if (weights == null || weights.length != this.weights.length || weights[0].length != this.weights[0].length) {
            throw new IllegalArgumentException("New weights dimensions do not match layer dimensions.");
        }
        this.weights = weights;
    }

    public void setBiases(double[] biases) {
        if (biases == null || biases.length != this.biases.length) {
            throw new IllegalArgumentException("New biases dimensions do not match layer dimensions.");
        }
        this.biases = biases;
    }
}