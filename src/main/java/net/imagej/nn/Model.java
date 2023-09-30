package net.imagej.nn;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import net.imagej.nn.enums.LayerType;
import net.imagej.nn.layers.Conv2D;
import net.imagej.nn.layers.Dense;
import net.imagej.nn.layers.Erosion2D;
import net.imagej.nn.layers.Flatten;
import net.imagej.nn.layers.MaxPooling2D;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Model {

    private List<Layer> layers = new ArrayList<>();

    public Model() {
    }

    public Model(List<Layer> layers) {
        this.layers = layers;
    }

    public void add(Layer layer) {
        layers.add(layer);
    }

    private JsonArray jsonLayers(String jsonFile) {
        JsonArray jsonArray;
        try (FileReader reader = new FileReader(jsonFile)) {
            jsonArray = new Gson().fromJson(reader, JsonArray.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return jsonArray;
    }

    public void load(String jsonFile) {
        JsonArray jsonArray = jsonLayers(jsonFile);
        int jsonIdx = 0;
        for (Layer layer : layers) {
            LayerType layerType = LayerType.getEnum(layer.getClass().getSimpleName());
            layer.load(jsonArray.get(jsonIdx).getAsJsonObject());
            switch (layerType) {
                case CONV_2D: case DILATION_2D: case DENSE: case EROSION_2D:
                    jsonIdx++;
                    break;
                case FLATTEN: case MAX_POOLING_2D:
                    break;
                default:
                    throw new IllegalStateException("Unexpected value: " + layerType);
            }
        }
    }


    public Object exec(Object input) {
        Object output = null;
        for (Layer layer : layers) {
            output = layer.exec(input);
            input = output;
        }
        return output;
    }
}
