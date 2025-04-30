package com.mlp;

import java.util.Arrays;
import java.util.Random;

import com.mlp.ActivationFunction.ActivationFunc;

public class Layer {

    private int numInputs;
    private int numOutputs;
    private double[][] weights;
    private double[] biases;

    private ActivationFunc activationFunction;

    private double[][] lastInput;
    private double[][] weigthWithBias;
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

        // --- Use WeightInit parameter ---
        initializeWeightsAndBiases(initMethod);

        // Initialize gradient storage
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
                    biases[i] = 0.0;
                }
                break;
        }
    }

    public double[][] forward(double[][] inputs) {
        if (inputs == null || inputs.length != 1 || inputs[0].length != numInputs) {
            throw new IllegalArgumentException("Input matrix dimensions incorrect. Expected [1][" + numInputs + "]");
        }
        this.lastInput = inputs;

        double[][] weightsTransposed = Matrix.transpose(this.weights);

        double[][] weightedSum = Matrix.multiply(this.lastInput, weightsTransposed);

        this.weigthWithBias = Matrix.addBiasVectorToRows(weightedSum, this.biases);
        this.activatedData = Matrix.applyFunc(this.weigthWithBias, this.activationFunction::activate);

        return this.activatedData;
    }

    public double[][] backward(double[][] deltaFromNextLayer, double[][] weightsFromNextLayer) {
        double[][] errorSignalPropagated;

        if (weightsFromNextLayer == null) {
            this.delta = deltaFromNextLayer;
        } else {
            errorSignalPropagated = Matrix.multiply(deltaFromNextLayer, weightsFromNextLayer);

            double[][] activationDerivative = Matrix.applyFunc(this.activatedData, activationFunction::derivative);

            this.delta = Matrix.multiplyElementWise(errorSignalPropagated, activationDerivative);
        }
        double[][] deltaTransposed = Matrix.transpose(this.delta);
        this.weightGradients = Matrix.multiply(deltaTransposed, this.lastInput);

        if (this.delta != null && this.delta.length > 0) {
            this.biasGradients = Arrays.copyOf(this.delta[0], this.delta[0].length);
        } else {
            this.biasGradients = new double[numOutputs];
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

    public double[] getBias() {
        return this.getBias();
    }

    public double[][] getWeightsGradient() {
        return this.weightGradients;
    }

    public double[] getBiasGradient() {
        return this.biasGradients;
    }

    public void setWeights(double[][] weights) {
        this.weights = weights;
    }

    public void setBiases(double[] biases) {
        this.biases = biases;
    }
}