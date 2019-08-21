package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.math.DifferentiableFunction;

import java.util.ArrayList;

//maximizing entropy by minimizing objective function: negative conditional log likelihood
//improve with softmax-margin
public class MyDifferentiableFunctionImpl implements DifferentiableFunction {
    double[] initialWeights;
    KbestListAndGoldTreeDataStructure kbestListAndGoldTreeDataStructure;
    ArrayList<int[]> kbestListFeatureVectors;
    int[] goldTreeFeatureVector;


    MyDifferentiableFunctionImpl(double[] weights, KbestListAndGoldTreeDataStructure kbestListAndGoldTreeDataStructure) {
        this.initialWeights = weights;
        this.kbestListAndGoldTreeDataStructure = kbestListAndGoldTreeDataStructure;
        this.kbestListFeatureVectors = kbestListAndGoldTreeDataStructure.getKbestListFeatureVectors();
        this.goldTreeFeatureVector = kbestListAndGoldTreeDataStructure.getGoldTreeFeatureVector();
    }
    @Override
    public double[] derivativeAt(double[] weights) {
        double[] gradients = new double[]{-1};

        return gradients;
    }

    @Override
    public double valueAt(double[] weights) {
        double result = 0;
        double L2 = L2of(weights);
        for(int i = 0; i < weights.length; i++) {

        }


        return result;
    }

    @Override
    public int dimension() {
        int dimension = initialWeights.length;
        return dimension;
    }

    private double L2of(double[] weights) {
        double result = 0;
        for(int i = 0; i < weights.length; i++) {
            result+=Math.pow(weights[i], 2);
        }
        result = Math.sqrt(result);
        return result;
    }
}
