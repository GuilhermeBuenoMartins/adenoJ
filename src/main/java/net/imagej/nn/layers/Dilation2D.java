package net.imagej.nn.layers;

import net.imagej.nn.Ops;
import net.imagej.nn.enums.Activation;

import java.util.Arrays;

public class Dilation2D {

    private double[][][][] input;

    private double[][][][] weights;

    private int[] strides;

    private Activation activation;

    private double[][][][] output;

    private double[] dilation(double[][][] window) {
        double[] dil = new double[output[0][0][0].length];
        double[][] max = new double[output[0][0][0].length][window[0][0].length];
        for (int i = 0; i < window.length; i++) { // Iterate number of rows
            for (int j = 0; j < window[0].length; j++) { // Iterate number of columns
                for (int k = 0; k < weights[0][0][0].length; k++) { // Iterate number of filters
                    double[] sum = Ops.add(window[i][j], Ops.t(weights[i][j])[k]);
                    for (int l = 0; l < max[k].length; l++) { // Iterate number of channels
                        max[k][l] = i == 0 && j == 0? sum[l]: Math.max(max[k][l], sum[l]);
                    }
                }
            }
        }
        for (int i = 0; i < dil.length; i++) {
            dil[i] = Arrays.stream(max[i]).sum();
        }
        return dil;
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
                    output[i][j][k] = dilation(window);
                }
            }
        }
    }

    private void reluActivation() {
        for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(i, j * strides[0], k * strides[1]);
                    output[i][j][k] = Ops.relu(dilation(window));
                }
            }
        }
    }

    private void sigmoidActivation() {
                for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(i, j * strides[0], k * strides[1]);
                    output[i][j][k] = Ops.sigmoid(dilation(window));
                }
            }
        }
    }

    private void applyConvolution() {
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
        applyConvolution();
        return output;
    }

}