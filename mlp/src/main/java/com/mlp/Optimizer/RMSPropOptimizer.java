package com.mlp.Optimizer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.mlp.Layer;

public class RMSPropOptimizer implements Optimizer {
    private double learningRate;
    private double beta2;
    private double epsilon;

    private Map<Layer, double[][]> cacheWeights;
    private Map<Layer, double[]> cacheBiases;

    public RMSPropOptimizer(double learningRate, double beta, double epsilon) {
        if (learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be positive.");
        if (beta2 < 0 || beta2 >= 1)
            throw new IllegalArgumentException("Beta2 must be in [0, 1).");
        if (epsilon <= 0)
            throw new IllegalArgumentException("Epsilon must be positive.");
        this.learningRate = learningRate;
        this.beta2 = beta;
        this.epsilon = epsilon;
        this.cacheWeights = new HashMap<>();
        this.cacheBiases = new HashMap<>();

    }

    public RMSPropOptimizer(double learningRate) {
        this(learningRate, 0.999, 1e-8);
    }

    @Override
    public void update(List<Layer> layers) {
        for (Layer layer : layers) {
            double[][] w = layer.getWeights();
            double[] b = layer.getBias();
            double[][] dw = layer.getWeightsGradient();
            double[] db = layer.getBiasGradient();

            if (w == null || b == null || dw == null || db == null)
                continue;

            double[][] sw = cacheWeights.computeIfAbsent(layer, k -> new double[w.length][w[0].length]);
            double[] sb = cacheBiases.computeIfAbsent(layer, k -> new double[b.length]);

            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[0].length; j++) {
                    sw[i][j] = this.beta2 * sw[i][j] + (1.0 - this.beta2) * (dw[i][j] * dw[i][j]);
                    w[i][j] -= this.learningRate * dw[i][j] / (Math.sqrt(sw[i][j]) + this.epsilon);
                }
            }

            for (int i = 0; i < b.length; i++) {
                sb[i] = this.beta2 * sb[i] + (1.0 - this.beta2) * (db[i] * db[i]);
                b[i] -= this.learningRate * db[i] / (Math.sqrt(sb[i]) + this.epsilon);
            }
        }
    }
}