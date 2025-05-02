package com.mlp;

public interface LossFunction {

    enum TaskType {
        REGRESSION,
        BINARY_CLASSIFICATION,
        MULTICLASS_CLASSIFICATION
    }

    double compute(double[][] predicted, double[][] target);

    double[][] derivative(double[][] predicted, double[][] target);

    class MeanSquaredError implements LossFunction {
        @Override
        public double compute(double[][] predicted, double[][] target) {
            if (predicted.length != target.length || predicted[0].length != target[0].length) {
                throw new IllegalArgumentException("Prediction and target dimensions must match.");
            }
            double sumSquaredError = 0.0;
            int numSamples = predicted.length;
            int numOutputs = predicted[0].length;

            for (int i = 0; i < numSamples; i++) {
                for (int j = 0; j < numOutputs; j++) {
                    double error = predicted[i][j] - target[i][j];
                    sumSquaredError += error * error;
                }
            }
            return sumSquaredError / (numSamples * numOutputs);
        }

        @Override
        public double[][] derivative(double[][] predicted, double[][] target) {
            return Matrix.substract(predicted, target);
        }
    }

    class CrossEntropyLoss implements LossFunction {
        private static final double EPSILON = 1e-15;

        @Override
        public double compute(double[][] predicted, double[][] target) {
            if (predicted.length != target.length || predicted[0].length != target[0].length) {
                throw new IllegalArgumentException("Prediction and target dimensions must match.");
            }
            double loss = 0.0;
            int numSamples = predicted.length;
            int numClasses = predicted[0].length;

            for (int i = 0; i < numSamples; i++) {
                for (int j = 0; j < numClasses; j++) {
                    double p = Math.max(EPSILON, Math.min(1.0 - EPSILON, predicted[i][j]));
                    loss -= target[i][j] * Math.log(p);
                    if (numClasses == 1) {
                        loss -= (1.0 - target[i][j]) * Math.log(1.0 - p);
                    } else if (numClasses > 1 && target[i][j] == 0) {
                    }
                }
            }
            return loss / numSamples;
        }

        @Override
        public double[][] derivative(double[][] predicted, double[][] target) {
            return Matrix.substract(predicted, target);
        }
    }
}