package com.mlp;

import com.mlp.ActivationFunction.*;
import com.mlp.LossFunction.*;
import com.mlp.Optimizer.*;

import java.io.IOException;

public class Main {

    private static final String MNIST_TRAIN_CSV_PATH = "mnist_train.csv";
    private static final String MNIST_TEST_CSV_PATH = "mnist_test.csv";
    private static final boolean HAS_HEADER = true;
    private static final String CSV_DELIMITER = ",";

    private static final int NUM_CLASSES = 10;
    private static final int INPUT_FEATURES = 28 * 28;

    private static final int MAX_EPOCHS = 10;
    private static final double LEARNING_RATE = 0.05;
    private static final int PATIENCE = 3;
    private static final double STOP_LOSS_THRESHOLD = 0.1;
    private static final int LOAD_LIMIT = 10000;

    public static void main(String[] args) {
        System.out.println("=== Starting MNIST MLP Test (CSV Version) ===");

        try {
            System.out.println("Loading MNIST training data...");
            DataPair trainData = CsvLoader.loadCsvData(MNIST_TRAIN_CSV_PATH, NUM_CLASSES, INPUT_FEATURES, CSV_DELIMITER,
                    HAS_HEADER, LOAD_LIMIT);
            System.out.printf("Loaded %d training samples.\n", trainData.features.length);

            System.out.println("Loading MNIST test data (for validation)...");
            int validationLimit = (LOAD_LIMIT > 0 && LOAD_LIMIT < 60000) ? LOAD_LIMIT / 6 : 2000;
            if (LOAD_LIMIT == -1)
                validationLimit = -1;
            DataPair validationData = CsvLoader.loadCsvData(MNIST_TEST_CSV_PATH, NUM_CLASSES, INPUT_FEATURES,
                    CSV_DELIMITER, HAS_HEADER, validationLimit);
            System.out.printf("Loaded %d validation samples.\n", validationData.features.length);

            validateData(trainData, validationData);

            System.out.println("\nConfiguring MLP model...");
            MLP mnistMlp = setupMLPModel();

            System.out.println("\nStarting Training...");
            long startTime = System.currentTimeMillis();
            mnistMlp.train(
                    trainData.features, trainData.labels,
                    validationData.features, validationData.labels,
                    MAX_EPOCHS,
                    PATIENCE,
                    STOP_LOSS_THRESHOLD);
            long endTime = System.currentTimeMillis();
            System.out.printf("Training completed in %.2f seconds.\n", (endTime - startTime) / 1000.0);

            System.out.println("\nEvaluating final model on full test set...");
            DataPair fullTestData;
            if (validationLimit != -1) {
                System.out.println("(Reloading full test set for final evaluation)");
                fullTestData = CsvLoader.loadCsvData(MNIST_TEST_CSV_PATH, NUM_CLASSES, INPUT_FEATURES, CSV_DELIMITER,
                        HAS_HEADER, -1);
            } else {
                fullTestData = validationData;
            }
            evaluate(mnistMlp, fullTestData);

            System.out.println("\n--- Testing prediction on individual samples ---");
            if (fullTestData != null && fullTestData.features.length > 0) {
                int[] sampleIndices = { 0, 42, 101, 500, 4 };

                for (int index : sampleIndices) {
                    if (index >= fullTestData.features.length) {
                        System.out.printf("Skipping index %d (out of bounds).\n", index);
                        continue;
                    }

                    double[] imageToTest = fullTestData.features[index];
                    int trueLabel = findIndexOfMax(fullTestData.labels[index]);

                    int predictedLabel = predictSingleImage(mnistMlp, imageToTest);

                    System.out.printf("Sample Index: %d | True Label: %d | Predicted Label: %d %s\n",
                            index,
                            trueLabel,
                            predictedLabel,
                            (trueLabel == predictedLabel ? "(Correct)" : "(Incorrect)"));

                }
            } else {
                System.out.println("Skipping single image prediction (no test data available).");
            }

        } catch (IOException e) {
            System.err.println("!!! ERROR reading CSV data. Check file paths, format, delimiter, and header setting.");
            e.printStackTrace();
        } catch (NumberFormatException e) {
            System.err.println("!!! ERROR parsing number from CSV. Check data integrity and delimiter.");
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("!!! An error occurred during MLP setup or training.");
            e.printStackTrace();
        }

        System.out.println("\n=== MNIST MLP Test (CSV Version) Completed ===");
    }

    private static MLP setupMLPModel() {
        int[] layerSizes = { INPUT_FEATURES, 128, 64, NUM_CLASSES };
        Optimizer optimizer = new SGDOptimizer(LEARNING_RATE);
        TaskType taskType = TaskType.MULTICLASS_CLASSIFICATION;
        WeighInit weighInit = WeighInit.HE_UNIFORM;
        ActivationFunc hiddenActivation = new ReLU();

        return new MLP(optimizer, taskType, weighInit, hiddenActivation, layerSizes);
    }

    private static void validateData(DataPair trainData, DataPair validationData) {
        if (trainData.features.length != trainData.labels.length
                || validationData.features.length != validationData.labels.length) {
            throw new RuntimeException("Data loading mismatch: Features and labels counts differ.");
        }
        if (trainData.features.length == 0) {
            throw new RuntimeException("No training data loaded. Check file path, format, and LOAD_LIMIT.");
        }
        if (validationData.features.length == 0) {
            System.out.println("Warning: No validation data loaded.");
        }
        if (trainData.features[0].length != INPUT_FEATURES) {
            throw new RuntimeException("Training image feature count mismatch. Expected " + INPUT_FEATURES);
        }
        if (validationData.features.length > 0 && validationData.features[0].length != INPUT_FEATURES) {
            throw new RuntimeException("Validation image feature count mismatch. Expected " + INPUT_FEATURES);
        }
        if (trainData.labels[0].length != NUM_CLASSES) {
            throw new RuntimeException("Training label dimension mismatch. Expected " + NUM_CLASSES);
        }
        if (validationData.labels.length > 0 && validationData.labels[0].length != NUM_CLASSES) {
            throw new RuntimeException("Validation label dimension mismatch. Expected " + NUM_CLASSES);
        }
    }

    public static void evaluate(MLP mlp, DataPair testData) {
        double[][] testImages = testData.features;
        double[][] testLabelsOneHot = testData.labels;

        if (testImages.length != testLabelsOneHot.length) {
            System.err.println("Evaluation error: Image and label counts mismatch.");
            return;
        }
        if (testImages.length == 0) {
            System.out.println("Evaluation skipped: No test data provided.");
            return;
        }

        int correctCount = 0;
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < testImages.length; i++) {
            double[][] inputSample = Matrix.rowVectorToMatrix(testImages[i]);
            double[][] prediction = mlp.forward(inputSample);
            int predictedClass = findIndexOfMax(prediction[0]);
            int trueClass = findIndexOfMax(testLabelsOneHot[i]);

            if (predictedClass == trueClass) {
                correctCount++;
            }
        }
        long endTime = System.currentTimeMillis();
        double accuracy = (double) correctCount / testImages.length;
        System.out.printf("Evaluation Accuracy: %.4f (%d / %d correct)\n", accuracy, correctCount, testImages.length);
        System.out.printf("Evaluation completed in %.2f seconds.\n", (endTime - startTime) / 1000.0);
    }

    public static int findIndexOfMax(double[] array) {
        if (array == null || array.length == 0)
            return -1;
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static int predictSingleImage(MLP mlp, double[] imageSample) {
        if (imageSample == null || imageSample.length != INPUT_FEATURES) {
            System.err.println("Error: Invalid image sample provided for prediction.");
            return -1;
        }

        double[][] inputMatrix = Matrix.rowVectorToMatrix(imageSample);

        double[][] predictionProbabilities = mlp.forward(inputMatrix);

        int predictedClass = findIndexOfMax(predictionProbabilities[0]);

        return predictedClass;
    }
}