package com.mlp;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        Matrix a = new Matrix();
        
        double[][] test = {{1,7,8},{2,4,5}};
        double[][] test2 = {{7,6,5}, {9,2,3}};
        double[][] dotProduct = a.add(test, test2);
        System.out.println(Arrays.deepToString(dotProduct));
    }
}