package net.imagej.nn.layers;

import java.util.Arrays;
import java.util.stream.IntStream;

import com.google.gson.JsonObject;

import net.imagej.nn.Layer;
import net.imagej.nn.Ops;
import net.imagej.nn.enums.Padding;

public class MaxPooling2D implements Layer {

    private int[] poolSize = new int[]{2, 2};

    private int[] strides = new int[]{1, 1};

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

    public MaxPooling2D(int[] poolSize, Padding padding) {
        this.poolSize = poolSize;
        this.padding = padding;
    }

    public MaxPooling2D(int[] poolSize, int[] strides, Padding padding) {
        this.poolSize = poolSize;
        this.strides = strides;
        this.padding = padding;
    }

    private double[][][][] padding(double[][][][] input) {
        int[] inputDims = new int[] {input.length, input[0].length, input[0][0].length, input[0][0][0].length};
        double[][][][] padInput = Ops.getPadInput(inputDims, poolSize);
        int[] padding = Ops.getPadding(poolSize);
        int padRow = Math.round((padding[0] - 1) / 2);
        int padCol = Math.round((padding[1] - 1) / 2);
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

    private double[][][] getWindow(double[][][][] input, int i, int initJ, int initK) {
        double[][][] window = new double[poolSize[0]][poolSize[1]][input[0][0][0].length];
        for (int j = 0; j < window.length; j++) {
            for (int k = 0; k < window[0].length; k++) {
                System.arraycopy(input[i][initJ + j][initK + k], 0, window[j][k], 0, input[i][initJ + j][initK + k].length);
            }
        }
        return window;
    }

    private double[] maxPool(double[][][] window) {
        double[] max = Arrays.stream(new double[window[0][0].length]).map(i -> Double.NEGATIVE_INFINITY).toArray();
        for (int j = 0; j < window.length; j++) { // Iterate number of rows
            double[] colsMax = Arrays.stream(Ops.t(window[j])).mapToDouble(r -> Arrays.stream(r).max().orElse(r[0])).toArray();
            double[] maxHist = max;
            max = IntStream.range(0, max.length).mapToDouble(i -> Math.max(maxHist[i], colsMax[i])).toArray();
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


    @Override
    public void load(JsonObject jsonObject) {
        // Layer without weights or bias
        // Unnecessary implementation
    }

    

    @Override
    public Object exec(Object input) {
        return exec((double[][][][]) input);
    }

    public double[][][][] exec(double[][][][] input) {
        input = padding.equals(Padding.SAME)? padding(input): input;
        int[] inputDims = new int[]{input[0].length, input[0][0].length, input[0][0][0].length};
        int[] paddingDims = new int[]{0, 0};
        int[] outputDims = Ops.getOutputDims(inputDims, poolSize, paddingDims, strides);
        output = new double[input.length][outputDims[0]][outputDims[1]][outputDims[2]];
        applyMaxPooling(input);
        return output;
    }

}
