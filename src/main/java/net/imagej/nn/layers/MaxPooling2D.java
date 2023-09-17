package net.imagej.nn.layers;

import net.imagej.nn.Layer;
import net.imagej.nn.Ops;

public class MaxPooling2D implements Layer {

    private double[][][][] input;

    private int[] poolSize;

    private int[] strides;

    private double[][][][] output;

    public MaxPooling2D() {
        this.poolSize = new int[]{2, 2};
        this.strides = new int[]{2, 2};
    }

    public MaxPooling2D(int[] poolSize, int[] strides) {
        this.poolSize = poolSize;
        this.strides = strides;
    }

    private double[][][] getWindow(int i, int initJ, int initK) {
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
    private void applyMaxPooling() {
        for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(i, j * strides[0], k * strides[1]);
                    output[i][j][k] =  maxPool(window);
                }
            }
        }
    }

    public double[][][][] exec(double[][][][] input) {
        this.input = input;
        int[] inputDims = new int[]{input[0].length, input[0][0].length, input[0][0][0].length};
        int[] paddingDims = new int[]{0, 0};
        int[] outputDims = Ops.getDim(inputDims, poolSize, paddingDims, strides);
        output = new double[input.length][outputDims[0]][outputDims[1]][outputDims[2]];
        applyMaxPooling();
        return output;
    }

}
