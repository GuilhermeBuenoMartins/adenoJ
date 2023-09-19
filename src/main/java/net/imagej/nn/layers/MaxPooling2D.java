package net.imagej.nn.layers;

import java.util.Arrays;

import net.imagej.nn.Layer;
import net.imagej.nn.Ops;
import net.imagej.nn.enums.Padding;

public class MaxPooling2D implements Layer {

    private int[] poolSize;

    private int[] strides;

    private Padding padding;

    private double[][][][] output;

    public MaxPooling2D() {
        this.poolSize = new int[]{2, 2};
        this.strides = new int[]{2, 2};
    }

    public MaxPooling2D(int[] poolSize, int[] strides) {
        this.poolSize = poolSize;
        this.strides = strides;
    }

    public MaxPooling2D(int[] poolSize, int[] strides, Padding padding) {
        this.poolSize = poolSize;
        this.strides = strides;
        this.padding = padding;
    }

    private double[][][][] padding(double[][][][] input) {
        int[] padding = new int[]{Math.round(poolSize[0] / 2), Math.round(poolSize[1] / 2)};
        double[][][][] padInput = new double[input.length][input[0].length + padding[0]][input[0][0].length + padding[1]][input[0][0][0].length];
        for (int i = 0; i < padInput.length; i++) {
            for (int j = 0; j < padInput[0].length; j++) {
                padInput[i][j] = Ops.apply(padInput[i][j], d -> d + Double.NEGATIVE_INFINITY);
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

    private double[][][] getWindow(double[][][][] input, int i, int initJ, int initK) {
        double[][][] window = new double[poolSize[0]][poolSize[1]][input[0][0][0].length];
        for (int j = 0; j < window.length; j++) {
            for (int k = 0; k < window[0].length; k++) {
                window[j][k] = input[i][initJ + j][initK + k];
            }
        }
        return window;
    }

    private double[] maxPool(double[][][] window) {
        double[] max = Ops.zeros(output[0][0][0].length);
        for (int j = 0; j < window.length; j++) { // Iterate number of rows
            for (int k = 0; k < window[0].length; k++) { //Iterate number of columns
                for (int l = 0; l < max.length; l++) { // Iterate channels
                    max[l] = j == 0 & k == 0? window[j][k][l]: Math.max(window[j][k][l], max[l]); // Max value preserving channels
                }
            }
        }
        return max;
    }
    private void applyMaxPooling(double[][][][] input) {
        for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(input, i, j * strides[0], k * strides[1]);
                    output[i][j][k] =  maxPool(window);
                }
            }
        }
    }

    public double[][][][] exec(double[][][][] input) {
        input = padding.equals(Padding.SAME)? padding(input): input;
        int[] inputDims = new int[]{input[0].length, input[0][0].length, input[0][0][0].length};
        int[] paddingDims = new int[]{0, 0};
        int[] outputDims = Ops.getDim(inputDims, poolSize, paddingDims, strides);
        output = new double[input.length][outputDims[0]][outputDims[1]][outputDims[2]];
        applyMaxPooling(input);
        return output;
    }

}
