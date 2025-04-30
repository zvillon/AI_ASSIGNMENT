package com.mlp.LossFunction;

import com.mlp.Matrix;

public class MeanSquaredError implements LossFunction {
    @Override
    public double compute(double[][] predicted, double[][] target) {
        double totalSquaredError = 0;
        for (int i = 0; i < predicted.length; ++i) {
            for (int j = 0; j < predicted[0].length; ++j) {
                totalSquaredError += Math.pow(predicted[i][j] - target[i][j], 2);
            }
        }
        int nbrOfElement = predicted.length * predicted[0].length;
        return totalSquaredError / (2 * nbrOfElement);
    }

    @Override
    public double[][] derivative(double[][] predicted, double[][] target) {
        int nbrOfElement = predicted.length * predicted[0].length;
        double value = 1 / (double) nbrOfElement;

        double[][] differenceMatrix = Matrix.substract(predicted, target);
        double[][] gradientMatrix = Matrix.multiply(differenceMatrix, value);
        return gradientMatrix;
    }
}
