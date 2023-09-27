package net.imagej.nn.layers;

import java.util.Arrays;

import net.imagej.nn.Ops;
import net.imagej.nn.enums.Activation;
import net.imagej.nn.enums.Padding;

public class Erosion2D {
    private double[][][][] weights;

    private int[] strides;

    private Activation activation;

    private Padding padding;

    private double[][][][] output;

    public Erosion2D(double[][][][] weights) {
        this.weights = weights;
        this.strides = new int[]{2, 2};
        this.activation = Activation.NONE;
        this.padding = Padding.SAME;
    }

    public Erosion2D(double[][][][] weights, int[] strides) {
        this.weights = weights;
        this.strides = strides;
        this.activation = Activation.NONE;
        this.padding = Padding.SAME;
    }

    public Erosion2D(double[][][][] weights, int[] strides, Activation activation) {
        this.weights = weights;
        this.strides = strides;
        this.activation = activation;
        this.padding = Padding.SAME;
    }

    public Erosion2D(double[][][][] weights, int[] strides, Activation activation, Padding padding) {
        this.weights = weights;
        this.strides = strides;
        this.activation = activation;
        this.padding = padding;
    }

    private double[][][][] padding(double[][][][] input) {
        int[] padding = new int[]{Math.round(weights.length / 2), Math.round(weights[0].length / 2)};
        double[][][][] padInput = new double[input.length][input[0].length + padding[0]][input[0][0].length + padding[1]][input[0][0][0].length];
        for (int i = 0; i < padInput.length; i++) {
            for (int j = 0; j < padInput[0].length; j++) {
                padInput[i][j] = Ops.apply(padInput[i][j], d -> d + Double.POSITIVE_INFINITY);
            }
        }
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) { // Reverse iteration of number of rows
                for (int k = 0; k < input[0][0].length; k++) { // Reverse iteration of number of rows
                    int padInputRow = padInput[0].length - 1 - j - padding[0];
                    int padInputCol = padInput[0][0].length - 1 - k - padding[1];
                    padInput[i][padInputRow][padInputCol] = input[i][input[0].length - 1 - j][input[0][0].length - 1 - k];
                }
            }
        }
        return padInput;
    }

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

    private double[][][] getWindow(double[][][][] input, int i, int initJ, int initK) {
        double[][][] window = new double[weights.length][weights[0].length][input[0][0][0].length];
        for (int j = 0; j < weights.length; j++) {
            for (int k = 0; k < weights[0].length; k++) {
                window[j][k] = input[i][initJ + j][initK + k];
            }
        }
        return window;
    }

    private void noneActivation(double[][][][] input) {
        for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(input, i, j * strides[0], k * strides[1]);
                    output[i][j][k] = erosion(window);
                }
            }
        }
    }

    private void reluActivation(double[][][][] input) {
        for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(input, i, j * strides[0], k * strides[1]);
                    output[i][j][k] = Ops.relu(erosion(window));
                }
            }
        }
    }

    private void sigmoidActivation(double[][][][] input) {
                for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(input, i, j * strides[0], k * strides[1]);
                    output[i][j][k] = Ops.sigmoid(erosion(window));
                }
            }
        }
    }

    private void applyErosion(double[][][][] input) {
        switch (activation) {
            case NONE:
                noneActivation(input);
                break;
            case RELU:
                reluActivation(input);
                break;
            case SIGMOID:
                sigmoidActivation(input);
        }
    }

    public double[][][][] exec(double[][][][] input) {
        input = padding.equals(Padding.SAME)? padding(input): input;
        int[] inputDims = new int[] { input[0].length, input[0][0].length, input[0][0][0].length };
        int[] weightsDims = new int[] { weights.length, weights[0].length };
        int numFilters = weights[0][0][0].length;
        int[] paddingDims = new int[] { 0, 0 };
        int[] outputDims = Ops.getDim(inputDims, weightsDims, numFilters, paddingDims, strides);
        System.out.printf("Output dims = %s \n", Arrays.toString(outputDims));
        output = new double[input.length][outputDims[0]][outputDims[1]][outputDims[2]];
        applyErosion(input);
        return output;
    }
    
}
