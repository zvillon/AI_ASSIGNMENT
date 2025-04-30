package com.mlp.Optimizer;

import java.util.List;

import com.mlp.Layer;

public interface Optimizer {
    void update(List<Layer> layers);
}
