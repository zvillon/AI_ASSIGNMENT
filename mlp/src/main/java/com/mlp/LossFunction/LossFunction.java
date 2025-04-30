package com.mlp.LossFunction;

public interface LossFunction {
    public double compute(double[][] predicted, double[][] target);

    public double[][] derivative(double[][] predicted, double[][] target);
}
