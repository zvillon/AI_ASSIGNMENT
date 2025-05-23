package com.cnn;

import java.util.Random;

public class ConvolutionLayer {

    private Random randomGenerator;

    private final int outputDepth;
    private final Output outputShape;
    private final Input inputShape;
    private final Kernel kernelShape;
    private final double[][][][] kernels;
    private final double[][][] biases;

    private double[][][] input;
    private double[][][] output;

    private class Output {

        private final int depth;
        private final int height;
        private final int width;

        public Output(int depth, int height, int width) {
            this.depth = depth;
            this.height = height;
            this.width = width;
        }

        public int getDepth() {
            return depth;
        }

        public int getHeight() {
            return height;
        }

        public int getWidth() {
            return width;
        }
    }

    private class Input {

        private final int depth;
        private final int height;
        private final int width;

        public Input(int depth, int height, int width) {
            this.depth = depth;
            this.height = height;
            this.width = width;
        }

        public int getDepth() {
            return depth;
        }

        public int getHeight() {
            return height;
        }

        public int getWidth() {
            return width;
        }
    }

    private class Kernel {
        private final int depth;
        private final int inputDepth;
        private final int size;

        public Kernel(int depth, int inputDepth, int kernelSize) {
            this.depth = depth;
            this.inputDepth = inputDepth;
            this.size = kernelSize;
        }

        public int getDepth() {
            return depth;
        }

        public int getInputDepth() {
            return inputDepth;
        }

        public int getSize() {
            return size;
        }
    }

    public ConvolutionLayer(int inputHeight, int inputWidth, int inputDepth, int kernelSize, int outputDepth) {
        this.outputDepth = outputDepth;
        this.inputShape = new Input(inputDepth, inputHeight, inputWidth);

        this.kernelShape = new Kernel(outputDepth, inputDepth, kernelSize);
        this.kernels = new double[outputDepth][inputDepth][kernelSize][kernelSize];

        for (int d = 0; d < outputDepth; d++) {
            for (int id = 0; id < inputDepth; id++) {
                for (int r = 0; r < kernelSize; r++) {
                    for (int c = 0; c < kernelSize; c++) {
                        kernels[d][id][r][c] = randomGenerator.nextGaussian();
                    }
                }
            }
        }

        int outputHeight = inputHeight - kernelSize + 1;
        int outputWidth = inputWidth - kernelSize + 1;
        this.outputShape = new Output(outputDepth, outputHeight, outputWidth);

        this.biases = new double[outputDepth][outputHeight][outputWidth];
        for (int d = 0; d < outputDepth; d++) {
            for (int r = 0; r < outputHeight; r++) {
                for (int c = 0; c < outputWidth; c++) {
                    biases[d][r][c] = randomGenerator.nextGaussian();
                }
            }
        }
    }

    public double[][][] forward(double[][][] input) {
        this.input = input;
        this.output = new double[outputDepth][outputShape.getHeight()][outputShape.getWidth()];
        for (int d = 0; d < outputDepth; d++) {
            for (int r = 0; r < outputShape.getHeight(); r++) {
                for (int c = 0; c < outputShape.getWidth(); c++) {
                    this.output[d][r][c] = biases[d][r][c];
                }
            }
        }

        int inputDepth = this.inputShape.getDepth();

        for (int od = 0; od < outputDepth; od++) {
            for (int id = 0; id < inputDepth; id++) {
                this.output[od] = correlateValid(input[id], kernels[od][id], this.output[od]);
            }
        }
        return this.output;
    }

    private double[][] correlateValid(double[][] inputMatrix, double[][] kernel2d, double[][] outputMatrix) {
        int kernelSize = kernel2d.length;
        int outputHeight = outputMatrix.length;
        int outputWidth = outputMatrix[0].length;

        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                for (int k = 0; k < kernelSize; k++) {
                    for (int l = 0; l < kernelSize; l++) {
                        outputMatrix[i][j] += inputMatrix[i + k][j + l] * kernel2d[k][l];
                    }
                }
            }
        }
        return outputMatrix;
    }
 
}
