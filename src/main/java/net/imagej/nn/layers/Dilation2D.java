package net.imagej.nn.layers;

import net.imagej.nn.Layer;
import net.imagej.nn.Ops;
import net.imagej.nn.enums.Activation;
import net.imagej.nn.enums.Padding;

import java.util.Arrays;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

public class Dilation2D implements Layer {

    private double[][][][] weights;

    private int[] strides = new int[] {1, 1};

    private Activation activation = Activation.NONE;

    private Padding padding = Padding.SAME;

    private double[][][][] output;

    public Dilation2D() {}

    public Dilation2D(Activation activation) {
        this.activation = activation;
    }

    public Dilation2D(Padding padding) {
        this.padding = padding;
    }

    public Dilation2D(Activation activation, Padding padding) {
        this.activation = activation;
        this.padding = padding;
    }


    public Dilation2D(double[][][][] weights) {
        this.weights = weights;
        this.strides = new int[]{2, 2};
        this.activation = Activation.NONE;
        this.padding = Padding.SAME;
    }

    public Dilation2D(double[][][][] weights, int[] strides) {
        this.weights = weights;
        this.strides = strides;
        this.activation = Activation.NONE;
        this.padding = Padding.SAME;
    }

    public Dilation2D(double[][][][] weights, int[] strides, Activation activation) {
        this.weights = weights;
        this.strides = strides;
        this.activation = activation;
        this.padding = Padding.SAME;
    }

    public Dilation2D(double[][][][] weights, int[] strides, Activation activation, Padding padding) {
        this.weights = weights;
        this.strides = strides;
        this.activation = activation;
        this.padding = padding;
    }

    private double[][][][] padding(double[][][][] input) {
        int[] inputDims = new int[] {input.length, input[0].length, input[0][0].length, input[0][0][0].length};
        int[] kernelDims = new int[] {weights.length, weights[0].length, weights[0][0].length, weights[0][0][0].length};
        double[][][][] padInput = Ops.getPadInput(inputDims, kernelDims);
        int[] padding = Ops.getPadding(kernelDims);
        int padRow = padding[0] / 2;
        int padCol = padding[1] / 2;
        for (int i = 0; i < padInput.length; i++) {
            for (int j = 0; j < padInput[0].length; j++) {
                padInput[i][j] = Ops.apply(padInput[i][j], d -> Double.NEGATIVE_INFINITY);
            }
        }
        for (int i = 0; i < input.length; i++) { // Iteration of number of samples
            for (int j = 0; j < input[0].length; j++) { // Iteration of number of rows
                System.arraycopy(input[i][j], 0, padInput[i][padRow + j], padCol, input[0][0].length);
                }
            }
        return padInput;
    }

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
                    output[i][j][k] = dilation(window);
                }
            }
        }
    }

    private void reluActivation(double[][][][] input) {
        for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(input, i, j * strides[0], k * strides[1]);
                    output[i][j][k] = Ops.relu(dilation(window));
                }
            }
        }
    }

    private void sigmoidActivation(double[][][][] input) {
                for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(input, i, j * strides[0], k * strides[1]);
                    output[i][j][k] = Ops.sigmoid(dilation(window));
                }
            }
        }
    }

    private void applyDilation(double[][][][] input) {
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

    public void load(JsonObject jsonObject) {
        Gson gson = new Gson();
        this.weights = gson.fromJson(jsonObject.get("weights"), double[][][][].class);
    }

    @Override
    public Object exec(Object input) {
        return exec((double[][][][]) input);
    }

    public double[][][][] exec(double[][][][] input) {
        input = padding.equals(Padding.SAME)? padding(input): input;
        int[] inputDims = new int[] { input[0].length, input[0][0].length, input[0][0][0].length };
        int[] weightsDims = new int[] { weights.length, weights[0].length };
        int numFilters = weights[0][0][0].length;
        int[] paddingDims = new int[] { 0, 0 };
        int[] outputDims = Ops.getOutputDims(inputDims, weightsDims, numFilters, paddingDims, strides);
        output = new double[input.length][outputDims[0]][outputDims[1]][outputDims[2]];
        applyDilation(input);
        return output;
    }

}