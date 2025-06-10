
package com.cnn;

public class LossFunction {

    public static double binaryCrossEntropy(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("yTrue and yPred must have the same length");
        }

        double sum = 0.0;
        int n = yTrue.length;

        for (int i = 0; i < n; i++) {

            double epsilon = 1e-15;
            double predClipped = Math.max(epsilon, Math.min(1.0 - epsilon, yPred[i]));

            sum += yTrue[i] * Math.log(predClipped) + (1.0 - yTrue[i]) * Math.log(1.0 - predClipped);
        }

        return -sum / n;
    }

    public static double[] binaryCrossEntropyPrime(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("yTrue and yPred must have the same length");
        }

        int n = yTrue.length;
        double[] gradient = new double[n];

        for (int i = 0; i < n; i++) {

            double epsilon = 1e-15;
            double predClipped = Math.max(epsilon, Math.min(1.0 - epsilon, yPred[i]));

            gradient[i] = ((1.0 - yTrue[i]) / (1.0 - predClipped) - yTrue[i] / predClipped) / n;
        }

        return gradient;
    }

    public static double binaryCrossEntropy(double[][] yTrue, double[][] yPred) {
        if (yTrue.length != yPred.length || yTrue[0].length != yPred[0].length) {
            throw new IllegalArgumentException("yTrue and yPred must have the same dimensions");
        }

        double sum = 0.0;
        int totalElements = yTrue.length * yTrue[0].length;

        for (int i = 0; i < yTrue.length; i++) {
            for (int j = 0; j < yTrue[0].length; j++) {
                double epsilon = 1e-15;
                double predClipped = Math.max(epsilon, Math.min(1.0 - epsilon, yPred[i][j]));

                sum += yTrue[i][j] * Math.log(predClipped) + (1.0 - yTrue[i][j]) * Math.log(1.0 - predClipped);
            }
        }

        return -sum / totalElements;
    }

    public static double[][] binaryCrossEntropyPrime(double[][] yTrue, double[][] yPred) {
        if (yTrue.length != yPred.length || yTrue[0].length != yPred[0].length) {
            throw new IllegalArgumentException("yTrue and yPred must have the same dimensions");
        }

        int rows = yTrue.length;
        int cols = yTrue[0].length;
        int totalElements = rows * cols;
        double[][] gradient = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double epsilon = 1e-15;
                double predClipped = Math.max(epsilon, Math.min(1.0 - epsilon, yPred[i][j]));

                gradient[i][j] = ((1.0 - yTrue[i][j]) / (1.0 - predClipped) - yTrue[i][j] / predClipped)
                        / totalElements;
            }
        }

        return gradient;
    }

    public static double binaryCrossEntropy(double[][][] yTrue, double[][][] yPred) {
        if (yTrue.length != yPred.length ||
                yTrue[0].length != yPred[0].length ||
                yTrue[0][0].length != yPred[0][0].length) {
            throw new IllegalArgumentException("yTrue and yPred must have the same dimensions");
        }

        double sum = 0.0;
        int totalElements = yTrue.length * yTrue[0].length * yTrue[0][0].length;

        for (int i = 0; i < yTrue.length; i++) {
            for (int j = 0; j < yTrue[0].length; j++) {
                for (int k = 0; k < yTrue[0][0].length; k++) {
                    double epsilon = 1e-15;
                    double predClipped = Math.max(epsilon, Math.min(1.0 - epsilon, yPred[i][j][k]));

                    sum += yTrue[i][j][k] * Math.log(predClipped)
                            + (1.0 - yTrue[i][j][k]) * Math.log(1.0 - predClipped);
                }
            }
        }

        return -sum / totalElements;
    }

    /**
     * Version pour matrices 3D - Binary Cross Entropy Prime
     */
    public static double[][][] binaryCrossEntropyPrime(double[][][] yTrue, double[][][] yPred) {
        if (yTrue.length != yPred.length ||
                yTrue[0].length != yPred[0].length ||
                yTrue[0][0].length != yPred[0][0].length) {
            throw new IllegalArgumentException("yTrue and yPred must have the same dimensions");
        }

        int depth = yTrue.length;
        int height = yTrue[0].length;
        int width = yTrue[0][0].length;
        int totalElements = depth * height * width;
        double[][][] gradient = new double[depth][height][width];

        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    double epsilon = 1e-15;
                    double predClipped = Math.max(epsilon, Math.min(1.0 - epsilon, yPred[i][j][k]));

                    gradient[i][j][k] = ((1.0 - yTrue[i][j][k]) / (1.0 - predClipped) - yTrue[i][j][k] / predClipped)
                            / totalElements;
                }
            }
        }

        return gradient;
    }

    public static void main(String[] args) {
        double[] yTrue = { 0.0, 1.0, 1.0, 0.0 };
        double[] yPred = { 0.1, 0.9, 0.8, 0.2 };

        double loss = binaryCrossEntropy(yTrue, yPred);
        double[] gradient = binaryCrossEntropyPrime(yTrue, yPred);

        System.out.println("Binary Cross Entropy Loss: " + loss);
        System.out.print("Gradient: [");
        for (int i = 0; i < gradient.length; i++) {
            System.out.print(gradient[i]);
            if (i < gradient.length - 1)
                System.out.print(", ");
        }
        System.out.println("]");
    }
}