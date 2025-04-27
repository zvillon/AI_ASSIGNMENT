package com.mlp.ActivationFunction;

public class Tanh implements ActivationFunc {

    @Override
    public double activate(double z) {
        return (Math.exp(z) - Math.exp(-z)) / (Math.exp(z) + Math.exp(-z));
    }

    @Override
    public double derivative(double z) {
        return 1 - Math.pow(activate(z), 2);
    }
}
