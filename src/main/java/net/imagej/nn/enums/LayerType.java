package net.imagej.nn.enums;

public enum LayerType {

    CONV_2D("Conv2D"),

    DILATION_2D("Dilation2D"),

    DENSE("Dense"),

    EROSION_2D("Erosion2D"),
    
    FLATTEN("Flatten"),

    MAX_POOLING_2D("MaxPooling2D");


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
            case "Dilation2D": return DILATION_2D;
            case "Dense": return DENSE;
            case "Erosion2D": return EROSION_2D;
            case "MaxPooling2D": return MAX_POOLING_2D;
            case "Flatten": return FLATTEN;
            default: throw new IllegalStateException("Layer type not found to value \"" + value + "\"");
        }
    }
}
