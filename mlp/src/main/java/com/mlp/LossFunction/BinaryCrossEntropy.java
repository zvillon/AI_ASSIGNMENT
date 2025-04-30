package com.mlp.LossFunction;

import com.mlp.Matrix;

public class BinaryCrossEntropy implements LossFunction{

    private final double epsilon = 1e-12;

    @Override
    public double compute(double[][] predicted, double[][] target) {
        double totalLoss = 0;
        for (int i = 0; i < predicted.length; ++i) {
            double clipped_a = Math.max(epsilon, Math.min((double)1 - epsilon, predicted[i][0]));
            totalLoss += (- (target[i][0] * Math.log(clipped_a) + (1 - (double)predicted[i][0]) * Math.log(1 - (double)clipped_a)));
        }
        return totalLoss / predicted.length;
    }
    
    @Override
    public double[][] derivative(double[][] predicted, double[][] target) {
        int batchSize = predicted.length;
        double[][] gradientMatrix = new double[batchSize][1];
        for (int i = 0; i < predicted.length; ++i) {
            double clipped_pred = Math.max(epsilon, Math.min(1 - (double)epsilon, predicted[i][0]));
            double denominator = clipped_pred *  (1 - (double)clipped_pred);
            double numerator = clipped_pred - target[i][0];
            if (Math.abs(denominator) < epsilon) {
                gradientMatrix[i][0] = (numerator > 0 ? 1.0 : -1.0) * 1.0 / epsilon / batchSize;
           } else {
                gradientMatrix[i][0] = (numerator / denominator) / batchSize;
           }
        }
        return gradientMatrix;
    }
}