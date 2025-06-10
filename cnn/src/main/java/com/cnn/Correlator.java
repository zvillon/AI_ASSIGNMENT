package com.cnn;

public class Correlator {

    static public double[][] correlateFull(double[][] inputMatrix, double[][] kernel2d) {
        return correlate(inputMatrix, kernel2d, "full");
    }

    static public double[][] correlateValid(double[][] inputMatrix, double[][] kernel2d) {
        return correlate(inputMatrix, kernel2d, "valid");
    }

    static private double[][] correlate(double[][] inputMatrix, double[][] kernel2d, String mode) {
        int inputHeight = inputMatrix.length;
        int inputWidth = inputMatrix[0].length;
        int kernelHeight = kernel2d.length;
        int kernelWidth = kernel2d[0].length;

        int padRows, padCols;

        int outputHeight, outputWidth;
        switch (mode.toLowerCase()) {
            case "full":
                outputHeight = inputHeight + kernelHeight - 1;
                outputWidth = inputWidth + kernelWidth - 1;
                padRows = kernelHeight - 1;
                padCols = kernelWidth - 1;
                break;
            case "valid":
                outputHeight = inputHeight - kernelHeight + 1;
                outputWidth = inputWidth - kernelWidth + 1;
                padRows = 0;
                padCols = 0;
                break;
            default:
                throw new IllegalArgumentException("Unsupported mode: " + mode);
        }

        double[][] outputMatrix = new double[outputHeight][outputWidth];

        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                double sum = 0.0;

                for (int k = 0; k < kernelHeight; k++) {
                    for (int l = 0; l < kernelWidth; l++) {

                        int inputRow = i - padRows + k;
                        int inputCol = j - padCols + l;

                        if (inputRow >= 0 && inputRow < inputHeight &&
                                inputCol >= 0 && inputCol < inputWidth) {
                            sum += inputMatrix[inputRow][inputCol] * kernel2d[k][l];
                        }
                    }
                }
                outputMatrix[i][j] = sum;
            }
        }

        return outputMatrix;
    }

    public double[][] convolveFull(double[][] inputMatrix, double[][] kernel2d) {
        double[][] flippedKernel = flipKernel(kernel2d);
        return correlate(inputMatrix, flippedKernel, "full");
    }

    public double[][] convolveSame(double[][] inputMatrix, double[][] kernel2d) {
        double[][] flippedKernel = flipKernel(kernel2d);
        return correlate(inputMatrix, flippedKernel, "same");
    }

    public double[][] convolveValid(double[][] inputMatrix, double[][] kernel2d) {
        double[][] flippedKernel = flipKernel(kernel2d);
        return correlate(inputMatrix, flippedKernel, "valid");
    }

    private double[][] flipKernel(double[][] kernel) {
        int height = kernel.length;
        int width = kernel[0].length;
        double[][] flippedKernel = new double[height][width];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                flippedKernel[i][j] = kernel[height - 1 - i][width - 1 - j];
            }
        }
        return flippedKernel;
    }
}