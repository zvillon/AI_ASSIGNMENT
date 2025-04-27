package com.mlp;

import java.util.Arrays;
import java.util.function.Function;

public class Matrix {

    public static double[][] transpose(double[][] a) {
        double[][] ret = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a[0].length; ++j) {
                ret[j][i] = a[i][j];
            }
        }
        return ret;
    }

    public static double[][] multiplyElementWise(double[][] a, double[][] b) {
        if (a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Matrices dimensions must be the same for element-wise multiplication.");
        }
        double[][] result = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a[0].length; ++j) {
                result[i][j] = a[i][j] * b[i][j];
            }
        }
        return result;
    }

    public static double[][] add(double[][] a, double[][] b) {
        double[][] ret = new double[a.length][a[0].length];
        if (a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Vectors dimensions are not the sames.");
        }
        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a[0].length; ++j) {
                ret[i][j] = a[i][j] + b[i][j];
            }
        }
        return ret;
    }

    public static double[][] substract(double[][] a, double[][] b) {
        double[][] ret = new double[a.length][a[0].length];
        if (a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Vectors dimensions are not the sames.");
        }
        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a[0].length; ++j) {
                ret[i][j] = a[i][j] - b[i][j];
            }
        }
        return ret;
    }

    public static double[][] applyFunc(double[][] a, Function<Double, Double> func) {
        double[][] ret = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a[0].length; ++j) {
                ret[i][j] = func.apply(a[i][j]);
            }
        }
        return ret;
    }

    public static double[][] addBiasVectorToRows(double[][] a, double[] bias) {
        double[][] ret = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; i < a[0].length; ++j) {
                ret[i][j] = a[i][j] + bias[j];
            }
        }
        return ret;
    }

    public static double dotProduct(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vector dimensions are not compatible for dot product.");
        }

        double result = 0;
        for (int i = 0; i < a.length; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    public static double[][] multiply(double[][] a, double[][] b) {
        // Just checking for errors in case matrix multiplications are possible or not, got those kind of error with numpy before
        if (a[0].length != b.length || b[0].length != a.length) {
            throw new IllegalArgumentException("Matrix dimensions are not compatible for multiplication.");
        }

        double[][] ret = new double[a.length][b[0].length];
        b = transpose(b);

        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a.length; ++j) {
                ret[i][j] = dotProduct(a[i], b[j]);
            }
        }

        return ret;
    }

    public static double[][] multiply(double[][] a, double scalar) {
        double[][] ret = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a[i].length; ++j) {
                ret[i][j] = a[i][j] * scalar;
            }
        }
        return ret;
    }

    public static double[] sumColumn(double[][] a) {
        double [] ret = new double[a.length];

        for (int i = 0; i < a[0].length; ++i) {
            double temps = 0;
            for (int j = 0; j < a.length; ++j) {
                temps += a[i][j];
            }
            ret[i] = temps;
        }
        return ret;
    }

    public Matrix() {

    }
}
