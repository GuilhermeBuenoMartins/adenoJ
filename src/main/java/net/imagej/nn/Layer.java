package net.imagej.nn;

import com.google.gson.JsonObject;

public interface Layer {

    void load(JsonObject jsonObject);

    Object exec(Object input);

}
