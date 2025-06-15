import java.util.*;
import java.io.*;
import java.nio.file.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

class ReLU {
    private double[][][] lastOutput;
    private double[] lastOutput1D;

    public double[][][] forward(double[][][] input) {
        lastOutput = new double[input.length][input[0].length][input[0][0].length];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                for (int k = 0; k < input[0][0].length; k++) {
                    lastOutput[i][j][k] = Math.max(0, input[i][j][k]);
                }
            }
        }
        return lastOutput;
    }

    public double[][][] backward(double[][][] outputGradient) {
        double[][][] inputGradient = new double[outputGradient.length][outputGradient[0].length][outputGradient[0][0].length];
        for (int i = 0; i < outputGradient.length; i++) {
            for (int j = 0; j < outputGradient[0].length; j++) {
                for (int k = 0; k < outputGradient[0][0].length; k++) {

                    if (lastOutput[i][j][k] > 0) {
                        inputGradient[i][j][k] = outputGradient[i][j][k];
                    } else {
                        inputGradient[i][j][k] = 0;
                    }
                }
            }
        }
        return inputGradient;
    }

    public double[] forward(double[] input) {
        lastOutput1D = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            lastOutput1D[i] = Math.max(0, input[i]);
        }
        return lastOutput1D;
    }

    public double[] backward(double[] outputGradient) {
        double[] inputGradient = new double[outputGradient.length];
        for (int i = 0; i < outputGradient.length; i++) {
            if (lastOutput1D[i] > 0) {
                inputGradient[i] = outputGradient[i];
            } else {
                inputGradient[i] = 0;
            }
        }
        return inputGradient;
    }
}

class ConvolutionLayer {
    private double[][][][] kernels;
    private double[] biases;
    private int inputWidth, inputHeight, inputDepth, numKernels, kernelSize;
    private Random random = new Random();

    private double[][][] lastInput;

    public ConvolutionLayer(int inputWidth, int inputHeight, int inputDepth, int kernelSize, int numKernels) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputDepth = inputDepth;
        this.kernelSize = kernelSize;
        this.numKernels = numKernels;

        kernels = new double[numKernels][inputDepth][kernelSize][kernelSize];
        biases = new double[numKernels];
        double limit = Math.sqrt(6.0 / (kernelSize * kernelSize * inputDepth + numKernels));

        for (int k = 0; k < numKernels; k++) {
            biases[k] = 0;
            for (int d = 0; d < inputDepth; d++) {
                for (int i = 0; i < kernelSize; i++) {
                    for (int j = 0; j < kernelSize; j++) {
                        kernels[k][d][i][j] = (random.nextDouble() * 2 - 1) * limit;
                    }
                }
            }
        }
    }

    public double[][][] forward(double[][][] input) {

        this.lastInput = input;

        int outputHeight = inputHeight - kernelSize + 1;
        int outputWidth = inputWidth - kernelSize + 1;
        double[][][] output = new double[numKernels][outputHeight][outputWidth];

        for (int k = 0; k < numKernels; k++) {
            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    double sum = 0.0;
                    for (int d = 0; d < inputDepth; d++) {
                        for (int ki = 0; ki < kernelSize; ki++) {
                            for (int kj = 0; kj < kernelSize; kj++) {
                                sum += input[d][i + ki][j + kj] * kernels[k][d][ki][kj];
                            }
                        }
                    }

                    output[k][i][j] = sum + biases[k];
                }
            }
        }
        return output;
    }

    public double[][][] backward(double[][][] outputGradient, double learningRate) {
        double[][][][] kernelsGradient = new double[numKernels][inputDepth][kernelSize][kernelSize];
        double[] biasesGradient = new double[numKernels];
        double[][][] inputGradient = new double[inputDepth][inputHeight][inputWidth];

        int outputHeight = outputGradient[0].length;
        int outputWidth = outputGradient[0][0].length;

        for (int k = 0; k < numKernels; k++) {
            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    double gradient = outputGradient[k][i][j];

                    biasesGradient[k] += gradient;

                    for (int d = 0; d < inputDepth; d++) {
                        for (int ki = 0; ki < kernelSize; ki++) {
                            for (int kj = 0; kj < kernelSize; kj++) {
                                kernelsGradient[k][d][ki][kj] += lastInput[d][i + ki][j + kj] * gradient;
                                inputGradient[d][i + ki][j + kj] += kernels[k][d][ki][kj] * gradient;
                            }
                        }
                    }
                }
            }
        }

        for (int k = 0; k < numKernels; k++) {
            biases[k] -= learningRate * biasesGradient[k];
            for (int d = 0; d < inputDepth; d++) {
                for (int i = 0; i < kernelSize; i++) {
                    for (int j = 0; j < kernelSize; j++) {
                        kernels[k][d][i][j] -= learningRate * kernelsGradient[k][d][i][j];
                    }
                }
            }
        }

        return inputGradient;
    }
}

class LossFunctions {
    public static double binaryCrossEntropy(double[] yTrue, double[] yPred) {
        double sum = 0.0;
        double epsilon = 1e-15;

        for (int i = 0; i < yTrue.length; i++) {
            double predClipped = Math.max(epsilon, Math.min(1.0 - epsilon, yPred[i]));
            sum += yTrue[i] * Math.log(predClipped) + (1.0 - yTrue[i]) * Math.log(1.0 - predClipped);
        }
        return -sum / yTrue.length;
    }
}

class Sigmoid {

    private double[][][] lastOutput3D;
    private double[] lastOutput1D;

    public double[][][] forward(double[][][] input) {
        lastOutput3D = new double[input.length][input[0].length][input[0][0].length];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                for (int k = 0; k < input[0][0].length; k++) {
                    lastOutput3D[i][j][k] = sigmoid(input[i][j][k]);
                }
            }
        }
        return lastOutput3D;
    }

    public double[][][] backward(double[][][] outputGradient) {
        double[][][] inputGradient = new double[outputGradient.length][outputGradient[0].length][outputGradient[0][0].length];
        for (int i = 0; i < outputGradient.length; i++) {
            for (int j = 0; j < outputGradient[0].length; j++) {
                for (int k = 0; k < outputGradient[0][0].length; k++) {
                    double s = lastOutput3D[i][j][k];
                    double derivative = s * (1 - s);
                    inputGradient[i][j][k] = outputGradient[i][j][k] * derivative;
                }
            }
        }
        return inputGradient;
    }

    public double[] forward(double[] input) {
        lastOutput1D = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            lastOutput1D[i] = sigmoid(input[i]);
        }
        return lastOutput1D;
    }

    public double[] backward(double[] outputGradient) {
        double[] inputGradient = new double[outputGradient.length];
        for (int i = 0; i < outputGradient.length; i++) {
            double s = lastOutput1D[i];
            double derivative = s * (1 - s);
            inputGradient[i] = outputGradient[i] * derivative;
        }
        return inputGradient;
    }

    private double sigmoid(double x) {

        return 1.0 / (1.0 + Math.exp(-Math.max(-250, Math.min(250, x))));
    }
}

class Dense {
    private double[][] weights;
    private double[] biases;
    private double[] lastInput;
    private Random random = new Random();

    public Dense(int inputSize, int outputSize) {

        double limit = Math.sqrt(6.0 / (inputSize + outputSize));
        weights = new double[inputSize][outputSize];
        biases = new double[outputSize];

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = (random.nextDouble() * 2 - 1) * limit;
            }
        }

        for (int i = 0; i < outputSize; i++) {
            biases[i] = 0.0;
        }
    }

    public double[] forward(double[] input) {
        this.lastInput = input.clone();
        double[] output = new double[biases.length];

        for (int j = 0; j < output.length; j++) {
            output[j] = biases[j];
            for (int i = 0; i < input.length; i++) {
                output[j] += input[i] * weights[i][j];
            }
        }
        return output;
    }

    public double[] backward(double[] outputGradient, double learningRate) {
        double[] inputGradient = new double[lastInput.length];
        double[][] weightsGradient = new double[weights.length][weights[0].length];

        for (int j = 0; j < outputGradient.length; j++) {
            for (int i = 0; i < lastInput.length; i++) {
                weightsGradient[i][j] = lastInput[i] * outputGradient[j];
                inputGradient[i] += weights[i][j] * outputGradient[j];
            }
        }

        for (int j = 0; j < outputGradient.length; j++) {
            biases[j] -= learningRate * outputGradient[j];
            for (int i = 0; i < lastInput.length; i++) {
                weights[i][j] -= learningRate * weightsGradient[i][j];
            }
        }

        return inputGradient;
    }
}

public class MNISTTraining {

    private ConvolutionLayer conv1;

    private ReLU relu1;
    private Dense dense1;
    private ReLU relu2;
    private Dense dense2;
    private Sigmoid sigmoid_final;

    public MNISTTraining() {

        conv1 = new ConvolutionLayer(28, 28, 1, 3, 5);
        relu1 = new ReLU();

        dense1 = new Dense(5 * 26 * 26, 128);
        relu2 = new ReLU();

        dense2 = new Dense(128, 10);
        sigmoid_final = new Sigmoid();
    }

    public double[] forward(double[][][] input) {
        double[][][] conv_out = conv1.forward(input);
        double[][][] relu1_out = relu1.forward(conv_out);

        double[] flattened = MNISTUtils.flatten(relu1_out);

        double[] dense1_out = dense1.forward(flattened);
        double[] relu2_out = relu2.forward(dense1_out);

        double[] dense2_out = dense2.forward(relu2_out);
        double[] final_out = sigmoid_final.forward(dense2_out);

        return final_out;
    }

    public void backward(double[] yTrue, double[] yPred, double learningRate) {

        double[] grad = new double[yPred.length];
        for (int i = 0; i < yPred.length; i++) {
            grad[i] = yPred[i] - yTrue[i];
        }

        grad = dense2.backward(grad, learningRate);
        grad = relu2.backward(grad);
        grad = dense1.backward(grad, learningRate);

        double[][][] grad3D = MNISTUtils.reshape(grad, 5, 26, 26);
        grad3D = relu1.backward(grad3D);

        conv1.backward(grad3D, learningRate);
    }

    public static void main(String[] args) {
        System.out.println("=== MNIST Training with CNN ===");

        int epochs = 20;
        double learningRate = 0.01;
        int trainLimit = 3000;
        int testLimit = 200;

        System.out.println("🔍 Looking for MNIST data...");
        MNISTUtils.MNISTData trainData = MNISTUtils.loadMNIST(trainLimit);
        MNISTUtils.MNISTData testData = MNISTUtils.loadMNIST(testLimit);

        System.out.printf("✅ Loaded data: %d training images, %d test images%n",
                trainData.images.length, testData.images.length);

        MNISTTraining network = new MNISTTraining();

        System.out.println("🚀 Starting training phase...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0.0;
            int correctInEpoch = 0;

            List<Integer> indices = IntStream.range(0, trainData.images.length).boxed().collect(Collectors.toList());
            Collections.shuffle(indices);

            int processed = 0;
            for (int idx : indices) {
                double[][][] input = new double[1][28][28];
                input[0] = trainData.images[idx];

                double[] yTrue = trainData.labels[idx];

                double[] yPred = network.forward(input);

                double error = LossFunctions.binaryCrossEntropy(yTrue, yPred);
                totalError += error;
                if (MNISTUtils.argmax(yPred) == MNISTUtils.argmax(yTrue)) {
                    correctInEpoch++;
                }

                network.backward(yTrue, yPred, learningRate);

                processed++;
                if (processed > 0 && processed % 500 == 0) {
                    System.out.printf("  Epoch %d: %d/%d samples processed...%n",
                            epoch + 1, processed, trainData.images.length);
                }
            }

            double avgError = totalError / trainData.images.length;
            double trainAccuracy = (double) correctInEpoch / trainData.images.length * 100;
            System.out.printf("✓ Epoch %d/%d, Average Error: %.6f, Training Accuracy: %.2f%%%n",
                    epoch + 1, epochs, avgError, trainAccuracy);
        }

        System.out.println("\n🧪 Evaluating the network...");
        int correct = 0;

        for (int i = 0; i < testData.images.length; i++) {
            double[][][] input = new double[1][28][28];
            input[0] = testData.images[i];

            double[] prediction = network.forward(input);
            int predClass = MNISTUtils.argmax(prediction);
            int trueClass = MNISTUtils.argmax(testData.labels[i]);

            if (predClass == trueClass) {
                correct++;
            }

            if (i < 15) {
                System.out.printf("  Sample %d - Prediction: %d, True: %d %s%n",
                        i, predClass, trueClass, (predClass == trueClass) ? "✓" : "✗");
            }
        }

        double accuracy = (double) correct / testData.images.length * 100;
        System.out.printf("\n🎯 Final Accuracy: %.2f%% (%d/%d)%n", accuracy, correct, testData.images.length);
    }
}