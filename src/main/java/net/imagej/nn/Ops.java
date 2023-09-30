package net.imagej.nn;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;
import java.util.stream.IntStream;

public class Ops {

    public static double[][] apply(double[][] a, DoubleUnaryOperator operator) {
        double[][] b = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            b[i] =  Arrays.stream(a[i]).map(operator).toArray();
        }
        return b;
    }

    public static double[] apply(double[] a, double[] b, IntToDoubleFunction function) {
        int aNumCols = a.length;
        int bNumCols = b.length;
        if (aNumCols != bNumCols) {
            String err = "Operation element-wise not allow. Expected number of matrix b dimensions equals to [%d], but was [%d].";
            throw new IndexOutOfBoundsException(String.format(err, aNumCols, bNumCols));
        }
        return IntStream.range(0, aNumCols).mapToDouble(function).toArray();
    }

    public static double[] add(double[] a, double[] b) {
        int aNumCols = a.length;
        int bNumCols = b.length;
        if (aNumCols != bNumCols) {
            String err = "Addition element-wise not allow. Expected number of matrix b dimensions equals to [%d], but was [%d].";
            throw new IndexOutOfBoundsException(String.format(err, aNumCols, bNumCols));
        }
        return apply(a, b, i -> a[i] + b[i]);
    }

    public static double[] sub(double[] a, double[] b) {
        int aNumCols = a.length;
        int bNumCols = b.length;
        if (aNumCols != bNumCols) {
            String err = "Addition element-wise not allow. Expected number of matrix b dimensions equals to [%d], but was [%d].";
            throw new IndexOutOfBoundsException(String.format(err, aNumCols, bNumCols));
        }
        return apply(a, b, i -> a[i] - b[i]);
    }

    public static double[][] add(double[][] a, double[][] b) {
        int aNumRows = a.length;
        int bNumRows = b.length;
        if (aNumRows != bNumRows) {
            String err = "Subtraction element-wise not allow. Expected number of matrix b dimensions equals to [%d], but was [%d].";
            throw new IndexOutOfBoundsException(String.format(err, aNumRows, bNumRows));
        }
        double[][] c = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            c[i] = add(a[i], b[i]);
        }
        return c;
    }

    public static double[][] sub(double[][] a, double[][] b) {
        int aNumRows = a.length;
        int bNumRows = b.length;
        if (aNumRows != bNumRows) {
            String err = "Subtraction element-wise not allow. Expected number of matrix b dimensions equals to [%d], but was [%d].";
            throw new IndexOutOfBoundsException(String.format(err, aNumRows, bNumRows));
        }
        double[][] c = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            c[i] = sub(a[i], b[i]);
        }
        return c;
    }

    public static double[][] dot(double[][] a, double[][] b) {
        int aNumCols = a[0].length;
        int bNumRows = b.length;
        if (aNumCols != bNumRows) {
            String err = "Matrix multiplication not allow. Expected number of matrix b rows equals to %d, but was %d.";
            throw new IndexOutOfBoundsException(String.format(err, bNumRows, aNumCols));
        }
        int aNumRows = a.length;
        int bNumCols = b[0].length;
        double[][] c = new double[aNumRows][bNumCols];
        for (int i = 0; i < aNumRows; i++) {
            for (int j = 0; j < bNumCols; j++) {
                int finalJ = j;
                int finalI = i;
                c[i][j] = IntStream.range(0, aNumCols).mapToDouble(k -> a[finalI][k] * b[k][finalJ]).sum();
            }
        }
        return c;
    }

    public static double[][] t(double[][] a) {
        double[][] b = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                b[j][i] = a[i][j];
            }
        }
        return b;
    }

    public static double[] sigmoid(double[] values) {
        return Arrays.stream(values).map(x -> 1 / (1 + Math.exp(-x))).toArray();
    }

    public static double[] relu(double[] values) {
        return Arrays.stream(values).map(x -> Math.max(x, 0)).toArray();
    }

    public static int[] getOutputDims(int[] inputDims, int[] kernelDims, int numFilters, int[] paddingDims, int[] strides) {
        int numRows = Math.floorDiv(inputDims[0] + 2 * paddingDims[0] - kernelDims[0], strides[0]) + 1;
        int numCols = Math.floorDiv(inputDims[1] + 2 * paddingDims[1] - kernelDims[1], strides[1]) + 1;
        int numChnl = numFilters;
        return new int[]{numRows, numCols, numChnl};
    }

    public static int[] getOutputDims(int[] inputDims, int[] poolSize, int[] paddingDims, int[] strides) {
        int numRows = Math.floorDiv(inputDims[0] + 2 * paddingDims[0] - poolSize[0], strides[0]) + 1;
        int numCols = Math.floorDiv(inputDims[1] + 2 * paddingDims[1] - poolSize[1], strides[1]) + 1;
        int numChnl = inputDims[2];
        return new int[]{numRows, numCols, numChnl};
    }

    public static int[] getPadding(int[] kernelDims) {
        return new int[] {kernelDims[0] - 1, kernelDims[1] - 1};
    }

    public static double[][][][] getPadInput(int[] inputDims, int[] kernelDims) {
        int[] padding = getPadding(kernelDims);
        return new double[inputDims[0]][inputDims[1] + padding[0]][ inputDims[2] + padding[1]][inputDims[3]];
    }

    public static double[] zeros(int widith) {
        return Arrays.stream(new double[widith]).map(i -> 0).toArray();
    }

    public static double[] ones(int widith) {
        return Arrays.stream(new double[widith]).map(i -> 1).toArray();
    }

    public static double[][][] zeros(int height, int width, int deepth) {
        double[][][] a = new double[height][width][deepth];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                a[i][j] = Arrays.stream(a[i][j]).map(elem -> 0).toArray();
            }
        }
        return a;
    }
}
