package net.imagej.nn.layers;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import net.imagej.nn.Layer;
import net.imagej.nn.Ops;
import net.imagej.nn.enums.Activation;

public class Dense implements Layer {

    private double[][] input;

    private double[][] weights;

    private double[] bias;

    private Activation activation;

    private double[][] output;

    public Dense(Activation activation) {
        this.activation = activation;
    }

    private void noneActivation() {
        output = Ops.dot(input, weights);
        for (int i = 0; i < input.length; i++) {
                output[i] = Ops.add(output[i], bias);
        }
    }

    private void reluActivation() {
        output = Ops.dot(input, weights);
        for (int i = 0; i < input.length; i++) {
            output[i] = Ops.relu(Ops.add(output[i], bias));
        }
    }

    private void sigmoidActivation() {
        output = Ops.dot(input, weights);
        for (int i = 0; i < input.length; i++) {
            output[i] = Ops.sigmoid(Ops.add(output[i], bias));
        }
    }

    private void applyActivation() {
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
        this.weights = gson.fromJson(jsonObject.get("weights"), double[][].class);
        this.bias = gson.fromJson(jsonObject.get("bias"), double[].class);
    }

    @Override
    public Object exec(Object input) {
        return exec((double[][]) input);
    }

    public double[][] exec(double[][] input) {
        this.input = input;
        output = new double[input.length][weights[0].length];
        applyActivation();
        return output;
    }

}
