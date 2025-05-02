package com.mlp.Optimizer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.mlp.Layer;

public class MomentumOptimizer implements Optimizer {

    private double learningRate;
    private double momentum;

    private Map<Layer, double[][]> velocityWeights;
    private Map<Layer, double[]> velocityBiases;

    public MomentumOptimizer(double learningRate, double momentum) {
        if (learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be positive.");
        if (momentum < 0 || momentum >= 1)
            throw new IllegalArgumentException("Beta must be in [0, 1).");
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.velocityBiases = new HashMap<>();
        this.velocityWeights = new HashMap<>();
    }

    @Override
    public void update(List<Layer> layers) {
        for (Layer layer : layers) {
            double[][] w = layer.getWeights();
            double[] b = layer.getBiases();
            double[][] dw = layer.getWeightsGradient();
            double[] db = layer.getBiasGradient();

            if (w == null || b == null || dw == null || db == null)
                continue;

            double[][] vw = velocityWeights.computeIfAbsent(layer,
                    k -> new double[w.length][w[0].length]);
            double[] vb = velocityBiases.computeIfAbsent(layer,
                    k -> new double[b.length]);

            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[0].length; j++) {
                    vw[i][j] = this.momentum * vw[i][j] + this.learningRate * dw[i][j];
                    w[i][j] -= vw[i][j];
                }
            }

            for (int i = 0; i < b.length; i++) {
                vb[i] = this.momentum * vb[i] + this.learningRate * db[i];
                b[i] -= vb[i];
            }

        }
    }
}