package net.imagej.nn.layers;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import net.imagej.nn.Layer;
import net.imagej.nn.Ops;
import net.imagej.nn.enums.Activations;

public class Conv2D implements Layer {

    private double[][][][] input;

    private double[][][][] weights;

    private double[] bias;

    private int[] strides;
    private Activations activation;

    private double[][][][] output;

    public Conv2D() {
        this.strides = new int[]{1, 1};
        this.activation = Activations.NONE;
    }

    public Conv2D(int[] strides, Activations activation) {
        this.strides = strides;
        this.activation = activation;
    }

    private double[] conv(double[][][] window) {
        double[] conv = Ops.zeros(output[0][0][0].length);
        for (int j = 0; j < window.length; j++) { // Iterate number of rows
            double[][] mul = new double[weights.length][weights[0].length];
            for (int k = 0; k < window[0].length; k++) { //Iterate number of columns
                mul[k] = Ops.dot(window[j], weights[j][k])[k]; // Multiply channels by filters
                for (int l = 0; l < weights[0][0][0].length; l++) { // Iterate channels
                    conv[l] += mul[k][l]; // Sum preserving channels
                }
            }
        }
        return conv;
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
                    output[i][j][k] = Ops.add(conv(window), bias);
                }
            }
        }
    }

    private void reluActivation() {
        for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(i, j * strides[0], k * strides[1]);
                    output[i][j][k] = Ops.relu(Ops.add(conv(window), bias));
                }
            }
        }
    }

    private void sigmoidActivation() {
        for (int i = 0; i < input.length; i++) { // Iterate number of samples
            for (int j = 0; j < output[0].length; j++) { // Iterate number of rows
                for (int k = 0; k < output[0][0].length; k++) { // Iterate number of columns
                    double[][][] window = getWindow(i, j * strides[0], k * strides[1]);
                    output[i][j][k] = Ops.sigmoid(Ops.add(conv(window), bias));
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

    public void load(JsonObject jsonObject) {
        Gson gson = new Gson();
        this.weights = gson.fromJson(jsonObject.get("weights"), double[][][][].class);
        this.bias = gson.fromJson(jsonObject.get("bias"), double[].class);
    }

    public double[][][][] exec(double[][][][] input) {
        this.input = input;
        int[] inputDims = new int[]{input[0].length, input[0][0].length, input[0][0][0].length};
        int[] weightsDims = new int[]{weights.length, weights[0].length};
        int numFilters =  weights[0][0][0].length;
        int[] paddingDims = new int[]{0, 0};
        int[] outputDims = Ops.getDim(inputDims, weightsDims, numFilters, paddingDims, strides);
        output = new double[input.length][outputDims[0]][outputDims[1]][outputDims[2]];
        applyConvolution();
        return output;
    }

}
