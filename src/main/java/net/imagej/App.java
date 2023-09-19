package net.imagej;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.imagej.nn.Layer;
import net.imagej.nn.Model;
import net.imagej.nn.Ops;
import net.imagej.nn.enums.Activation;
import net.imagej.nn.enums.Padding;
import net.imagej.nn.layers.*;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Hello world!
 */
public class App {
        public static void main(String[] args) {
                double[][][][] a = new double[][][][] {
                                { { { 2.1293125, -1.1328828 },
                                                { -1.7876774, -0.22082105 },
                                                { 0.27193397, 0.19348656 } },
                                                { { 0.7172302, 0.15078504 },
                                                                { -0.7120794, -0.9116528 },
                                                                { 0.49493262, 0.11187045 } },
                                                { { 1.6110834, -1.2986826 },
                                                                { -0.19753598, 0.91646266 },
                                                                { 0.5573477, -1.4438905 } },
                                                { { -0.32141778, 0.00690729 },
                                                                { 1.3889844, -1.0410852 },
                                                                { 0.937285, -0.26477095 } },
                                                { { -1.6397643, 0.56324255 },
                                                                { -0.5628855, 1.2299346 },
                                                                { 1.5360107, -0.46731105 } },
                                                { { -0.40626192, 0.927702 },
                                                                { -0.02944215, 0.27145854 },
                                                                { 0.8905089, 0.31852916 } } },
                                { { { 0.10905274, -0.05732935 },
                                                { -0.52965385, -1.4799902 },
                                                { 1.4528421, 0.82369924 } },
                                                { { -0.37572452, -1.0227187 },
                                                                { 1.3036335, 0.3829994 },
                                                                { -0.39000285, 0.05549616 } },
                                                { { 1.5415694, 0.109445 },
                                                                { 0.49152088, -1.2140148 },
                                                                { -0.4016121, 1.6470238 } },
                                                { { -1.4579073, -0.7150865 },
                                                                { 1.062184, 0.37837905 },
                                                                { 0.07492701, 0.7735992 } },
                                                { { 0.64455324, 0.32997242 },
                                                                { 0.24127619, -0.55505943 },
                                                                { 0.5321247, 0.40671933 } },
                                                { { 1.3472501, -0.2933444 },
                                                                { 0.18362416, 0.8224893 },
                                                                { 0.5187017, -0.35494503 } } } };
                double[][][][] w = new double[][][][] {
                                {
                                                { { 0.06416305, -0.02356174 }, { 0.02466866, -0.01145447 } },
                                                { { 0.01566892, 0.03317402 }, { -0.02672133, -0.01867163 } } },
                                {
                                                { { -0.00756757, 0.02629718 }, { 0.02238744, 0.0419078 } },
                                                { { -0.03585156, 0.07758348 }, { 0.08057486, 0.03336884 } } } };
                double[] b = new double[] { 0.00, 0.00};
                int[] strides = new int[] { 1, 1 };
                double[][][][] c = new MaxPooling2D(new int[]{2, 2}, strides, Padding.SAME).exec(a);
                System.out.println(Arrays.deepToString(c));
                int[] cShape = new int[] {c.length, c[0].length, c[0][0].length, c[0][0][0].length};
                System.out.printf("\nShape: %s\n", Arrays.toString(cShape));

                // final String JSON_FILE =
                // "/home/gbueno/PycharmProjects/pythonProject/model.json";
                //
                // Model model = new Model();
                // model.add(new Conv2D());
                // model.add(new MaxPooling2D());
                // model.add(new Flatten());
                // model.add(new Dense(Activations.RELU));
                // model.add(new Dense(Activations.SIGMOID));
                //
                // model.load(JSON_FILE);
                //
                // double[][] output = (double[][]) model.exec(a);
                // System.out.println(Arrays.deepToString(output));
        }
}
