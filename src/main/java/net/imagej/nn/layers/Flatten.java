package net.imagej.nn.layers;

import com.google.gson.JsonObject;

import net.imagej.nn.Layer;

public class Flatten implements Layer {

    @Override
    public void load(JsonObject jsonObject) {
        // Layer without weights or bias
        // Unnecessary implementation
    }

    @Override
    public Object exec(Object input) {
        return exec((double[][][][]) input);
    }

    public double[][] exec(double[][][][] input) {
        int numFeatures = input[0].length * input[0][0].length * input[0][0][0].length;
        double[][] output = new double[input.length][numFeatures];
        for (int i = 0; i < output.length; i++) { // Iterate samples
            int feat = 0;
            for (int j = 0; j < input[0].length; j++) { // Iterate rows
                for (int k = 0; k < input[0][0].length; k++) { // Iterate columns
                    for (int l = 0; l < input[0][0][0].length; l++) { // Iterate channels
                        output[i][feat] = input[i][j][k][l];
                        feat++;
                    }
                }

            }
        }
        return output;
    }

}
