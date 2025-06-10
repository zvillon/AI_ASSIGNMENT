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

    public class Output {

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

    public class Input {

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
                this.output[od] = Correlator.correlateValid(input[id], kernels[od][id]);
            }
        }
        return this.output;
    }

    // private double[][] correlateValid(double[][] inputMatrix, double[][] kernel2d, double[][] outputMatrix) {
    //     int kernelSize = kernel2d.length;
    //     int outputHeight = outputMatrix.length;
    //     int outputWidth = outputMatrix[0].length;

    //     for (int i = 0; i < outputHeight; i++) {
    //         for (int j = 0; j < outputWidth; j++) {
    //             for (int k = 0; k < kernelSize; k++) {
    //                 for (int l = 0; l < kernelSize; l++) {
    //                     outputMatrix[i][j] += inputMatrix[i + k][j + l] * kernel2d[k][l];
    //                 }
    //             }
    //         }
    //     }
    //     return outputMatrix;
    // }

    private double[][][] backward(double[][][] outputGradient, double learningRate) {
        double[][][][] kernelsGradient = new double[this.outputDepth][this.kernelShape.inputDepth][this.kernelShape.depth][this.kernelShape.depth];
        for (int i = 0; i < this.outputDepth; ++i) {
            for (int j = 0; j < this.kernelShape.inputDepth; ++j) {
                for (int k = 0; k < this.kernelShape.depth; ++k) {
                    for (int l = 0; l < this.kernelShape.depth; ++l) {
                        kernelsGradient[i][j][k][l] = 0.0;
                    }
                }
            }
        }
        int inputDepth = this.inputShape.getDepth();
        int inputHeight = this.inputShape.getHeight();
        int inputWidth = this.inputShape.getWidth();
        double[][][] inputGradient = new double[inputDepth][inputHeight][inputWidth];
        for (int i = 0; i < inputDepth; ++i) {
            for (int j = 0; j < inputHeight; ++j) {
                for (int k = 0; k < inputWidth; ++k) {
                    inputGradient[i][j][k] = 0.0;
                }
            }
        }

        for (int i = 0; i < this.outputDepth; ++i) {
            for (int j = 0; j < inputDepth; ++j) {
                kernelsGradient[i][j] = Correlator.correlateValid(input[j], outputGradient[i]);
            }
        }

        for (int i = 0; i < this.outputDepth; ++i) {
            for (int j = 0; j < this.kernelShape.inputDepth; ++j) {
                for (int k = 0; k < this.kernelShape.depth; ++k) {
                    for (int l = 0; l < this.kernelShape.depth; ++l) {
                        this.kernels[i][j][k][l] -= learningRate * kernelsGradient[i][j][k][l];
                        inputGradient[j] = Correlator.correlateFull(outputGradient[i], this.kernels[i][j]);
                    }
                }
            }
        }

        for (int i = 0; i < this.outputDepth; ++i) {
            for (int j = 0; j < this.outputShape.getHeight(); ++j) {
                for (int k = 0; k < this.outputShape.getWidth(); ++k) {
                    this.biases[i][j][k] -= learningRate * outputGradient[i][j][k];

                }
            }
        }
        return inputGradient;
    }
 
}
