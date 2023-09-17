package net.imagej.nn.enums;

public enum LayerType {

    CONV_2D("Conv2D"),

    DENSE("Dense"),

    MAX_POOLING_2D("MaxPooling2D"),

    FLATTEN("Flatten");

    private String value;

    LayerType(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }


    public static LayerType getEnum(String value) {
        switch (value) {
            case "Conv2D": return CONV_2D;
            case "Dense": return DENSE;
            case "MaxPooling2D": return MAX_POOLING_2D;
            case "Flatten": return FLATTEN;
            default: throw new IllegalStateException("Layer type not found to value \"" + value + "\"");
        }
    }
}
