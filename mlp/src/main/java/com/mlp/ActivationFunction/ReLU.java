package com.mlp.ActivationFunction;

public class ReLU implements ActivationFunc {
    
    @Override
    public double activate(double z) {
        return Math.max(0, z);
    }

    @Override
    public double derivative(double z) {
        return z > 0 ? 1 : 0;
    }
}
