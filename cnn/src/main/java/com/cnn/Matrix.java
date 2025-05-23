package com.cnn;

public class Matrix {

    private double[][] data;
    private int[] shape;

    public Matrix() {

    }

    public int[] getShape() {
        return this.shape;
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

}