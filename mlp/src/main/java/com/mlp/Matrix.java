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

    public static double[][] rowVectorToMatrix(double[] v) {
        double[][] matrix = new double[1][v.length];
        matrix[0] = Arrays.copyOf(v, v.length);
        return matrix;
    }

    public static double[][] multiply(double[][] a, double[][] b) {
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

    public static double[] sumColumns(double[][] a) {
        if (a == null || a.length == 0) return new double[0];
        int numCols = a[0].length;
        double[] columnSums = new double[numCols];
        for (int j = 0; j < numCols; j++) {
            for (int i = 0; i < a.length; i++) {
                columnSums[j] += a[i][j];
            }
        }
        return columnSums;
    }

    public static double[][] deepCopy(double[][] original) {
        if (original == null) return null;
        double[][] result = new double[original.length][];
        for (int i = 0; i < original.length; i++) {
            result[i] = Arrays.copyOf(original[i], original[i].length);
        }
        return result;
     }

    public Matrix() {

    }
}
