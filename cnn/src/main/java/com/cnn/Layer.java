package com.cnn;

import java.util.List;

public interface Layer {
    Matrix forward(Matrix input);
    Matrix backward(Matrix outputGradient);

    List<Matrix> getParameters();
    List<Matrix> getGradients();
    void updateParameters(List<Matrix> updates, double learningRate);
}