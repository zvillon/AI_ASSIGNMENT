package com.mlp.Optimizer;

import java.util.List;

import com.mlp.Layer;

public class SGDOptimizer implements Optimizer {

    private double learningRate;

    public SGDOptimizer(double learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.learningRate = learningRate;
    }

    @Override
    public void update(List<Layer> layers) {
        for (Layer layer: layers) {
            double[][] w = layer.getWeights();
            double[] b = layer.getBias();
            double[][] dw = layer.getWeightsGradient();
            double[] db = layer.getBiasGradient();

            if (w == null || b ==null || dw == null || db == null) {
                System.err.println("Warning: Gradients or params missing for a layer during update");
                continue;
            }

            for (int i = 0; i < w.length; ++i) {
                for (int j = 0; j < w[0].length; ++j) {
                    w[i][j] -= this.learningRate * dw[i][j];
                }
            }
            for (int i = 0; i < b.length; ++i) {
                b[i] -= this.learningRate * db[i];
            }
        }
    }
}