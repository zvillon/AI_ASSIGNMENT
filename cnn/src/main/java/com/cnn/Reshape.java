package com.cnn;

import com.cnn.ConvolutionLayer.Input;
import com.cnn.ConvolutionLayer.Output;

public class Reshape {
    Input input_shape;
    Output output_shape;

    public Reshape(Input input_shape, Output output_shape) {
        this.input_shape = input_shape;
        this.output_shape = output_shape;
    }

    public double[][][] forward(double[][][] input) {
        double[][][] output = new double[output_shape.getDepth()][output_shape.getHeight()][output_shape.getWidth()];
        for (int i = 0; i < input_shape.getDepth(); i++) {
            for (int j = 0; j < input_shape.getHeight(); j++) {
                for (int k = 0; k < input_shape.getWidth(); k++) {
                    output[i][j][k] = input[i][j][k];
                }
            }
        }
        return output;
    }

    public double[][][] backward(double[][][] output) {
        double[][][] input = new double[input_shape.getDepth()][input_shape.getHeight()][input_shape.getWidth()];
        for (int i = 0; i < output_shape.getDepth(); i++) {
            for (int j = 0; j < output_shape.getHeight(); j++) {
                for (int k = 0; k < output_shape.getWidth(); k++) {
                    input[i][j][k] = output[i][j][k];
                }
            }
        }
        return input;
    }
}