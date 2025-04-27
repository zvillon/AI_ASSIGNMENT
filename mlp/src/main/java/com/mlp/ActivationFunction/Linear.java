package com.mlp.ActivationFunction;

public class Linear implements ActivationFunc {

    @Override
    public double activate(double z) {
        return z;
    }

    @Override
    public double derivative(double z) {
        return 1;
    }
}
