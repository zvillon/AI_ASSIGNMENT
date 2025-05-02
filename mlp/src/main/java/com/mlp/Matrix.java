package com.mlp;

import java.util.Arrays;
import java.util.function.Function;

public class Matrix {

    private Matrix() {
    }

    public static double[][] transpose(double[][] a) {
        if (a == null || a.length == 0 || a[0].length == 0) {
            throw new IllegalArgumentException("Input matrix cannot be null or empty for transpose.");
        }
        int rows = a.length;
        int cols = a[0].length;
        double[][] ret = new double[cols][rows];
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                ret[j][i] = a[i][j];
            }
        }
        return ret;
    }

    public static double[][] multiplyElementWise(double[][] a, double[][] b) {
        if (a == null || b == null || a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Matrices dimensions must be the same for element-wise multiplication.");
        }
        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result[i][j] = a[i][j] * b[i][j];
            }
        }
        return result;
    }

    public static double[][] add(double[][] a, double[][] b) {
        if (a == null || b == null || a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Matrices dimensions must be the same for addition.");
        }
        int rows = a.length;
        int cols = a[0].length;
        double[][] ret = new double[rows][cols];
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                ret[i][j] = a[i][j] + b[i][j];
            }
        }
        return ret;
    }

    public static double[][] substract(double[][] a, double[][] b) {
        if (a == null || b == null || a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Matrices dimensions must be the same for subtraction.");
        }
        int rows = a.length;
        int cols = a[0].length;
        double[][] ret = new double[rows][cols];
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                ret[i][j] = a[i][j] - b[i][j];
            }
        }
        return ret;
    }

    public static double[][] applyFunc(double[][] a, Function<Double, Double> func) {
        if (a == null)
            return null;
        int rows = a.length;
        if (rows == 0)
            return new double[0][0];
        int cols = a[0].length;
        double[][] ret = new double[rows][cols];
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                ret[i][j] = func.apply(a[i][j]);
            }
        }
        return ret;
    }

    public static double[][] addBiasVectorToRows(double[][] a, double[] bias) {
        if (a == null || bias == null || a.length == 0 || a[0].length != bias.length) {
            throw new IllegalArgumentException("Matrix columns must match bias vector length.");
        }
        int rows = a.length;
        int cols = a[0].length;
        double[][] ret = new double[rows][cols];
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                ret[i][j] = a[i][j] + bias[j];
            }
        }
        return ret;
    }

    public static double dotProduct(double[] a, double[] b) {
        if (a == null || b == null || a.length != b.length) {
            throw new IllegalArgumentException("Vector dimensions are not compatible for dot product.");
        }
        double result = 0;
        for (int i = 0; i < a.length; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    public static double[][] rowVectorToMatrix(double[] v) {
        if (v == null)
            return new double[1][0];
        double[][] matrix = new double[1][v.length];
        matrix[0] = Arrays.copyOf(v, v.length);
        return matrix;
    }

    public static double[][] multiply(double[][] a, double[][] b) {
        if (a == null || b == null || a.length == 0 || b.length == 0 || a[0].length != b.length) {
            throw new IllegalArgumentException(
                    String.format("Matrix dimensions are not compatible for multiplication: A(%d x %d), B(%d x %d)",
                            a.length, a[0].length, b.length, b[0].length));
        }

        int aRows = a.length;
        int aCols = a[0].length;
        int bCols = b[0].length;

        double[][] result = new double[aRows][bCols];

        for (int i = 0; i < aRows; ++i) {
            for (int j = 0; j < bCols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < aCols; ++k) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    public static double[][] multiply(double[][] a, double scalar) {
        if (a == null)
            return null;
        int rows = a.length;
        if (rows == 0)
            return new double[0][0];
        int cols = a[0].length;
        double[][] ret = new double[rows][cols];
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                ret[i][j] = a[i][j] * scalar;
            }
        }
        return ret;
    }

    public static double[] sumColumns(double[][] a) {
        if (a == null || a.length == 0)
            return new double[0];
        int numRows = a.length;
        int numCols = a[0].length;
        double[] columnSums = new double[numCols];
        for (int j = 0; j < numCols; j++) {
            for (int i = 0; i < numRows; i++) {
                if (a[i] != null && j < a[i].length) {
                    columnSums[j] += a[i][j];
                }
            }
        }
        return columnSums;
    }

    public static double sumRows(double[][] a) {
        if (a == null || a.length == 0)
            return 0;
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                sum += a[i][j];
            }
        }
        return sum;
    }

    public static double[][] deepCopy(double[][] original) {
        if (original == null)
            return null;
        double[][] result = new double[original.length][];
        for (int i = 0; i < original.length; i++) {
            if (original[i] != null) {
                result[i] = Arrays.copyOf(original[i], original[i].length);
            } else {
                result[i] = null;
            }
        }
        return result;
    }

    public static double[][] square(double[][] a) {
        if (a == null)
            return null;
        double[][] result = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; ++i)
            for (int j = 0; j < a[0].length; ++j)
                result[i][j] = a[i][j] * a[i][j];
        return result;
    }

    public static double[] square(double[] a) {
        if (a == null)
            return null;
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; ++i)
            result[i] = a[i] * a[i];
        return result;
    }

    public static double[][] sqrt(double[][] a) {
        if (a == null)
            return null;
        double[][] result = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; ++i)
            for (int j = 0; j < a[0].length; ++j)
                result[i][j] = Math.sqrt(a[i][j]);
        return result;
    }

    public static double[] sqrt(double[] a) {
        if (a == null)
            return null;
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; ++i)
            result[i] = Math.sqrt(a[i]);
        return result;
    }

    public static double[][] divide(double[][] a, double[][] b) {
        if (a == null || b == null || a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Matrices dimensions must be the same for division.");
        }
        double[][] result = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; ++i)
            for (int j = 0; j < a[0].length; ++j) {
                if (b[i][j] == 0)
                    throw new ArithmeticException("Division by zero.");
                result[i][j] = a[i][j] / b[i][j];
            }

        return result;
    }

    public static double[] divide(double[] a, double[] b) {
        if (a == null || b == null || a.length != b.length) {
            throw new IllegalArgumentException("Vector dimensions must be the same for division.");
        }
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; ++i) {
            if (b[i] == 0)
                throw new ArithmeticException("Division by zero.");
            result[i] = a[i] / b[i];
        }
        return result;
    }

}