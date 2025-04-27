package com.mlp.ActivationFunction;

public class Sigmoid implements ActivationFunc {

    @Override
    public double activate(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    @Override
    public double derivative(double z) {
        double activation = activate(z);
        return activation * (1 - activation);
    }
}
