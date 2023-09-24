package net.imagej.nn.layers;

import java.util.Arrays;

import net.imagej.nn.Ops;
import net.imagej.nn.enums.Activation;

public class Erosion2D {

    private double[][][][] input;

    private double[][][][] weights;

    private int[] strides;

    private Activation activation;

    private double[][][][] output;

    private double[] erosion(double[][][] window) {
        double[] ero = new double[output[0][0][0].length];
        double[][] min = new double[output[0][0][0].length][window[0][0].length];
        for (int i = 0; i < window.length; i++) { // Iterate number of rows
            for (int j = 0; j < window[0].length; j++) { // Iterate number of columns
                for (int k = 0; k < weights[0][0][0].length; k++) { // Iterate number of filters
                    double[] sub = Ops.sub(window[i][j], Ops.t(weights[weights.length - 1 - i][weights[0].length -1 - j])[k]);
                    for (int l = 0; l < min[k].length; l++) { // Iterate number of channels
                        min[k][l] = i == 0 && j == 0? sub[l]: Math.min(min[k][l], sub[l]);
                    }
                }
            }
        }
        for (int i = 0; i < ero.length; i++) {
            ero[i] = Arrays.stream(min[i]).sum();
        }
        return ero;
    }

    private double[][][] getWindow(int i, int initJ, int initK) {
        double[][][] window = new double[weights.length][weights[0].length][input[0][0][0].length];
        for (int j = 0; j < weights.length; j++) {
            for (int k = 0; k < weights[0].length; k++) {
                window[j][k] = input[i][initJ + j][initK + k];
            }
        }
        return window;
    }

    private void noneActivation() {
        for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(i, j * strides[0], k * strides[1]);
                    output[i][j][k] = erosion(window);
                }
            }
        }
    }

    private void reluActivation() {
        for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(i, j * strides[0], k * strides[1]);
                    output[i][j][k] = Ops.relu(erosion(window));
                }
            }
        }
    }

    private void sigmoidActivation() {
                for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(i, j * strides[0], k * strides[1]);
                    output[i][j][k] = Ops.sigmoid(erosion(window));
                }
            }
        }
    }

    private void applyErosion() {
        switch (activation) {
            case NONE:
                noneActivation();
                break;
            case RELU:
                reluActivation();
                break;
            case SIGMOID:
                sigmoidActivation();
        }
    }

    public double[][][][] exec(double[][][][] input, double[][][][] weigthts, int[] strides, Activation activation) {
        this.input = input;
        this.weights = weigthts;
        this.strides = strides;
        this.activation = activation;
        int[] inputDims = new int[] { input[0].length, input[0][0].length, input[0][0][0].length };
        int[] weightsDims = new int[] { weights.length, weights[0].length };
        int numFilters = weights[0][0][0].length;
        int[] paddingDims = new int[] { 0, 0 };
        int[] outputDims = Ops.getDim(inputDims, weightsDims, numFilters, paddingDims, strides);
        System.out.printf("Output dims = %s \n", Arrays.toString(outputDims));
        output = new double[input.length][outputDims[0]][outputDims[1]][outputDims[2]];
        applyErosion();
        return output;
    }
    
}
