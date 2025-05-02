package com.mlp.Optimizer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.mlp.Layer;

public class AdamOptimizer implements Optimizer {

    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;

    private final Map<Layer, double[][]> mWeights;
    private final Map<Layer, double[]> mBiases;
    private final Map<Layer, double[][]> vWeights;
    private final Map<Layer, double[]> vBiases;
    private int t;

    public AdamOptimizer(double learningRate, double beta1, double beta2, double epsilon) {
        if (learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be positive.");
        if (beta1 < 0 || beta1 >= 1)
            throw new IllegalArgumentException("Beta1 must be in [0, 1).");
        if (beta2 < 0 || beta2 >= 1)
            throw new IllegalArgumentException("Beta2 must be in [0, 1).");
        if (epsilon <= 0)
            throw new IllegalArgumentException("Epsilon must be positive.");
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.mWeights = new HashMap<>();
        this.mBiases = new HashMap<>();
        this.vWeights = new HashMap<>();
        this.vBiases = new HashMap<>();
        this.t = 0;
    }

    public AdamOptimizer(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    @Override
    public void update(List<Layer> layers) {
        t++;

        for (Layer layer : layers) {
            double[][] w = layer.getWeights();
            double[] b = layer.getBiases();
            double[][] dw = layer.getWeightsGradient();
            double[] db = layer.getBiasGradient();

            if (w == null || b == null || dw == null || db == null)
                continue;

            double[][] mw = mWeights.computeIfAbsent(layer, k -> new double[w.length][w[0].length]);
            double[] mb = mBiases.computeIfAbsent(layer, k -> new double[b.length]);
            double[][] vw = vWeights.computeIfAbsent(layer, k -> new double[w.length][w[0].length]);
            double[] vb = vBiases.computeIfAbsent(layer, k -> new double[b.length]);

            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[0].length; j++) {
                    mw[i][j] = this.beta1 * mw[i][j] + (1.0 - this.beta1) * dw[i][j];
                    vw[i][j] = this.beta2 * vw[i][j] + (1.0 - this.beta2) * (dw[i][j] * dw[i][j]);

                    double mw_corr = mw[i][j] / (1.0 - Math.pow(this.beta1, t));
                    double vw_corr = vw[i][j] / (1.0 - Math.pow(this.beta2, t));

                    w[i][j] -= this.learningRate * mw_corr / (Math.sqrt(vw_corr) + this.epsilon);
                }
            }

            for (int i = 0; i < b.length; i++) {
                mb[i] = this.beta1 * mb[i] + (1.0 - this.beta1) * db[i];
                vb[i] = this.beta2 * vb[i] + (1.0 - this.beta2) * (db[i] * db[i]);

                double mb_corr = mb[i] / (1.0 - Math.pow(this.beta1, t));
                double vb_corr = vb[i] / (1.0 - Math.pow(this.beta2, t));

                b[i] -= this.learningRate * mb_corr / (Math.sqrt(vb_corr) + this.epsilon);
            }
        }
    }
}