package com.mlp;

import java.util.Arrays;

public interface ActivationFunction {

    @FunctionalInterface
    interface ActivationFunc {
        double activate(double x);
        default double derivative(double activatedValue) {
            return 1.0;
        }
    }

    class Linear implements ActivationFunc {
        @Override
        public double activate(double x) {
            return x;
        }
        @Override
        public double derivative(double activatedValue) {
            return 1.0;
        }
    }

    class Sigmoid implements ActivationFunc {
        @Override
        public double activate(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
        @Override
        public double derivative(double activatedValue) {
            return activatedValue * (1.0 - activatedValue);
        }
    }

    class ReLU implements ActivationFunc {
        @Override
        public double activate(double x) {
            return Math.max(0, x);
        }
        @Override
        public double derivative(double activatedValue) {
            return activatedValue > 0 ? 1.0 : 0.0;
        }
    }

     class Softmax implements ActivationFunc {
         @Override
         public double activate(double x) {
            return x;
         }

         @Override
         public double derivative(double activatedValue) {
             return 1.0;
         }

         public static double[] activateVector(double[] inputVector) {
             double[] output = new double[inputVector.length];
             double maxVal = Arrays.stream(inputVector).max().orElse(0.0);
             double sumExp = 0.0;
             for (int i = 0; i < inputVector.length; i++) {
                 output[i] = Math.exp(inputVector[i] - maxVal);
                 sumExp += output[i];
             }
             if (sumExp == 0) sumExp = 1e-15;
             for (int i = 0; i < inputVector.length; i++) {
                 output[i] /= sumExp;
             }
             return output;
         }
         public static double[][] activateMatrix(double[][] inputMatrix) {
              double[][] outputMatrix = new double[inputMatrix.length][];
              for(int i=0; i<inputMatrix.length; ++i) {
                  outputMatrix[i] = activateVector(inputMatrix[i]);
              }
              return outputMatrix;
         }
    }
}