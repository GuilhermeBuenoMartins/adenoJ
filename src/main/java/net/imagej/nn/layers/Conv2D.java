package net.imagej.nn.layers;

import java.util.Arrays;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import net.imagej.nn.Layer;
import net.imagej.nn.Ops;
import net.imagej.nn.enums.Activation;
import net.imagej.nn.enums.Padding;

public class Conv2D implements Layer {

    private double[][][][] weights;

    private double[] bias;

    private int[] strides = new int[]{1, 1};

    private Padding padding = Padding.VALID;

    private Activation activation = Activation.NONE;

    private double[][][][] output;

    public Conv2D() {
        this.strides = new int[]{1, 1};
        this.activation = Activation.NONE;
    }

    public Conv2D(double[][][][] weights, double[] bias, int[] strides, Activation activation) {
        this.weights = weights;
        this.bias = bias;
        this.strides = strides;
        this.activation = activation;
    }

    public Conv2D(double[][][][] weights, double[] bias) {
        this.weights = weights;
        this.bias = bias;
        this.strides = new int[]{1, 1};
        this.activation = Activation.NONE;
        this.padding = Padding.SAME;
        
    }

    public Conv2D(Padding padding) {
        this.padding = padding;
    }

    public Conv2D(int[] strides, Padding padding){
        this.strides = strides;
        this.padding = padding;
    }

    public Conv2D(int[] strides, Activation activation) {
        this.strides = strides;
        this.activation = activation;
    }

    private double[][][][] padding(double[][][][] input) {
        int[] inputDims = new int[] {input.length, input[0].length, input[0][0].length, input[0][0][0].length};
        int[] kernelDims = new int[] {weights.length, weights[0].length, weights[0][0].length, weights[0][0][0].length};
        double[][][][] padInput = Ops.getPadInput(inputDims, kernelDims);
        int[] padding = Ops.getPadding(kernelDims);
        int padRow = Math.round((padding[0] - 1) / 2);
        int padCol = Math.round((padding[1] - 1) / 2);
        for (int i = 0; i < input.length; i++) { // Iteration of number of samples
            for (int j = 0; j < input[0].length; j++) { // Iteration of number of rows
                System.arraycopy(input[i][j], 0, padInput[i][padRow + j], padCol, input[0][0].length);
                }
            }
        return padInput;
    }

    private double[] conv(double[][][] window) {
        double[] conv = Ops.zeros(output[0][0][0].length);
        for (int j = 0; j < window.length; j++) { // Iterate number of rows
            double[][] mul = new double[weights.length][weights[0].length];
            for (int k = 0; k < window[0].length; k++) { //Iterate number of columns
                mul = Ops.dot(window[j], weights[j][k]); // Multiply channels by filters
                for (int l = 0; l < weights[0][0][0].length; l++) { // Iterate channels
                    conv[l] += mul[k][l]; // Sum preserving channels
                }
            }
        }
        return conv;
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
                    output[i][j][k] = Ops.add(conv(window), bias);
                }
            }
        }
    }

    private void reluActivation(double[][][][] input) {
        for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(input, i, j * strides[0], k * strides[1]);
                    output[i][j][k] = Ops.relu(Ops.add(conv(window), bias));
                }
            }
        }
    }

    private void sigmoidActivation(double[][][][] input) {
        for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(input, i, j * strides[0], k * strides[1]);
                    output[i][j][k] = Ops.sigmoid(Ops.add(conv(window), bias));
                }
            }
        }
    }

    private void applyConvolution(double[][][][] input) {
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
        this.bias = gson.fromJson(jsonObject.get("bias"), double[].class);
    }

    @Override
    public Object exec(Object input) {
        return exec((double[][][][]) input);
    }

    public double[][][][] exec(double[][][][] input) {
        if (padding.equals(Padding.SAME)) {
            input = padding(input);
        }
        int[] inputDims = new int[]{input[0].length, input[0][0].length, input[0][0][0].length};
        int[] weightsDims = new int[]{weights.length, weights[0].length};
        int numFilters =  weights[0][0][0].length;
        int[] paddingDims = new int[]{0, 0};
        int[] outputDims = Ops.getOutputDims(inputDims, weightsDims, numFilters, paddingDims, strides);
        output = new double[input.length][outputDims[0]][outputDims[1]][outputDims[2]];
        applyConvolution(input);
        return output;
    }

}
