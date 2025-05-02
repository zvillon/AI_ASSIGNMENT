package com.cnn;

public class ActivationFunction {

    @FunctionalInterface
    interface ActivationFunc {
        public double activate(double x);

        default public double derivative(double activatedValue) {
            return 1.0;
        }
    }

    public class LeakyReLU implements ActivationFunc {

        private final double alpha = 0.1;

        @Override
        public double activate(double x) {
            return Math.max(this.alpha * x, x);
        }

        @Override
        public double derivative(double activatedValue) {
            return activatedValue > 0 ? 1.0 : this.alpha;
        }
    }

    public class ReLU implements ActivationFunc {
        @Override
        public double activate(double x) {
            return Math.max(0, x);
        }

        @Override
        public double derivative(double activatedValue) {
            return activatedValue > 0 ? 1.0 : 0.0;
        }
    }
}