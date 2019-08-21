package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.math.DifferentiableFunction;

import java.util.ArrayList;

public class MyMathDifferentiableFunction implements DifferentiableFunction {
    private double[] initialWeights;
    private ArrayList<KbestListAndGoldTreeDataStructure> kbestListAndGoldTreeDataStructures;
    private double lambda = 0.01;
    private double[] scoreOfPseudoGoldTreeOfEachKbestList;
    private int[] argmaxIndices;
    private double[] sumOfScoreOfEachTreeInKbestList;
    private int timesCalled = 0;

    MyMathDifferentiableFunction(double[] initialWeights, ArrayList<KbestListAndGoldTreeDataStructure> kbestListAndGoldTreeDataStructures) {
        this.initialWeights = initialWeights;
        this.kbestListAndGoldTreeDataStructures = kbestListAndGoldTreeDataStructures;
        this.scoreOfPseudoGoldTreeOfEachKbestList = new double[kbestListAndGoldTreeDataStructures.size()];
        this.argmaxIndices = new int[kbestListAndGoldTreeDataStructures.size()];
        this.sumOfScoreOfEachTreeInKbestList = new double[kbestListAndGoldTreeDataStructures.size()];
    }

    @Override
    public double valueAt(double[] weights) {
        timesCalled++;
        if(timesCalled % 1000 == 0) {
            System.out.println("Done learning on " + timesCalled + " training pairs of kbestLists and gold trees.");
        } else if (timesCalled == 36765) {
            timesCalled = 0;
        }
        double likelihood;

        for(int i = 0; i < kbestListAndGoldTreeDataStructures.size(); i++) {
            double sumOfScoreOfEachTreeInThisKbestList = 0;
            KbestListAndGoldTreeDataStructure kbestListAndGoldTreeDataStructure = kbestListAndGoldTreeDataStructures.get(i);
            double best = Double.POSITIVE_INFINITY;
            int argmaxIndex = -1;
            ArrayList<Double> kbestListLosses = kbestListAndGoldTreeDataStructure.getKbestListLosses();
            for(int k = 0; k < kbestListAndGoldTreeDataStructure.getKbestListFeatureVectors().size(); k++) {
                double score = score(kbestListAndGoldTreeDataStructure, k, weights);
                sumOfScoreOfEachTreeInThisKbestList+=score;
                double loss = kbestListLosses.get(k);
                if(loss < best) {
                    best = loss;
                    argmaxIndex = k;
                }
            }
            scoreOfPseudoGoldTreeOfEachKbestList[i]=score(kbestListAndGoldTreeDataStructure, argmaxIndex, weights);
            sumOfScoreOfEachTreeInKbestList[i] = sumOfScoreOfEachTreeInThisKbestList;
            argmaxIndices[i] = argmaxIndex;
        }

        double scaledL2 = lambda * L2of(weights);
        likelihood = scaledL2;

        for(int i = 0; i < kbestListAndGoldTreeDataStructures.size(); i++) {
            likelihood-=(Math.log(scoreOfPseudoGoldTreeOfEachKbestList[i]) - Math.log(sumOfScoreOfEachTreeInKbestList[i]));
        }
        return likelihood;
    }

    @Override
    public double[] derivativeAt(double[] weights) {
        double[] gradients = new double[weights.length];

//        int[][] featuresOfBestOfEachKbestList = new int[kbestListAndGoldTreeDataStructures.size()][];
//        double[][] scoresOfEachTreeInKbestListTimesThatTreesFeatures = new double[kbestListAndGoldTreeDataStructures.size()][];
//        for(int i = 0; i < kbestListAndGoldTreeDataStructures.size(); i++) {
//            KbestListAndGoldTreeDataStructure kbestListAndGoldTreeDataStructure = kbestListAndGoldTreeDataStructures.get(i);
//            double best = Double.NEGATIVE_INFINITY;
//            int argmaxIndex = -1;
//            ArrayList<int[]> kbestListFeatureVectors = kbestListAndGoldTreeDataStructure.getKbestListFeatureVectors();
//            double[] sumOf_ScoresOfEachTreeInKbestListTimesThatTreesFeatures = new double[weights.length];
//            for (int k = 0; k < kbestListFeatureVectors.size(); k++) {
//                double score = score(kbestListAndGoldTreeDataStructure, k, weights);
//                int[] kbestListFeatureVector = kbestListFeatureVectors.get(k);
//                double[] scoredKbestListFeatureVector = new double[kbestListFeatureVector.length];
//                for(int h = 0; h < kbestListFeatureVector.length; h++) {
//                    scoredKbestListFeatureVector[h] = score * kbestListFeatureVector[h];
//                }
//                if (score > best) {
//                    best = score;
//                    argmaxIndex = k;
//                }
//                for(int j = 0; j < kbestListFeatureVector.length; j++) {
//                    sumOf_ScoresOfEachTreeInKbestListTimesThatTreesFeatures[kbestListFeatureVector[j]] += scoredKbestListFeatureVector[j];
//                }
//            }
//            featuresOfBestOfEachKbestList[i]=kbestListAndGoldTreeDataStructure.getKbestListFeatureVectors().get(argmaxIndex);
//            scoresOfEachTreeInKbestListTimesThatTreesFeatures[i] = sumOf_ScoresOfEachTreeInKbestListTimesThatTreesFeatures;
//        }
//
//        double[] differences = new double[weights.length];
//        for(int i = 0; i < kbestListAndGoldTreeDataStructures.size(); i++) {
//            int[] featuresOfBestOfThisKbestList = featuresOfBestOfEachKbestList[i];
//            double[] scoresOfEachTreeInThisKbestListTimesThatTreesFeatures = scoresOfEachTreeInKbestListTimesThatTreesFeatures[i];
//            //for j in length of featuresOfBestOfThisKbestList
//                //differences[featuresOfBestOfThisKbestList[j]] = featuresOfBestOfThisKbestList[j] - scoresOfEach...[featuresOfBestOfThisKbestList[j]]
//            for(int j = 0; j < featuresOfBestOfThisKbestList.length; j++) {
//                differences[featuresOfBestOfThisKbestList[j]] = featuresOfBestOfThisKbestList[j] - scoresOfEachTreeInThisKbestListTimesThatTreesFeatures[featuresOfBestOfThisKbestList[j]];
//            }
//        }
        double[] differences = new double[weights.length];

        for(int i = 0; i < kbestListAndGoldTreeDataStructures.size(); i++) {
            KbestListAndGoldTreeDataStructure kbestListAndGoldTreeDataStructure = kbestListAndGoldTreeDataStructures.get(i);
            double best = Double.NEGATIVE_INFINITY;
            int argmaxIndex = -1;
            ArrayList<int[]> kbestListFeatureVectors = kbestListAndGoldTreeDataStructure.getKbestListFeatureVectors();
            double[] sumOf_ScoresOfEachTreeInKbestListTimesThatTreesFeatures = new double[weights.length];

            for(int k = 0; k < kbestListFeatureVectors.size(); k++) {
                double score = score(kbestListAndGoldTreeDataStructure, k, weights);
                int[] kbestListFeatureVector = kbestListFeatureVectors.get(k);
                double[] scoredKbestListFeatureVector = new double[kbestListFeatureVector.length];
                for(int h = 0; h < kbestListFeatureVector.length; h++) {
                    scoredKbestListFeatureVector[h] = score * kbestListFeatureVector[h];
                }
                if (score > best) {
                    best = score;
                    argmaxIndex = k;
                }
                for(int j = 0; j < kbestListFeatureVector.length; j++) {
                    sumOf_ScoresOfEachTreeInKbestListTimesThatTreesFeatures[kbestListFeatureVector[j]] += scoredKbestListFeatureVector[j];
                }
            }
            int[] featuresOfPseudoGoldTreeOfThisKbestList = kbestListAndGoldTreeDataStructure.getKbestListFeatureVectors().get(argmaxIndices[i]);
//            int[] featuresOfBestOfThisKbestList = kbestListAndGoldTreeDataStructure.getKbestListFeatureVectors().get(argmaxIndex);

            for(int j = 0; j < featuresOfPseudoGoldTreeOfThisKbestList.length; j++) {
                differences[featuresOfPseudoGoldTreeOfThisKbestList[j]] += 1 - sumOf_ScoresOfEachTreeInKbestListTimesThatTreesFeatures[featuresOfPseudoGoldTreeOfThisKbestList[j]];
            }
        }

        for(int i = 0; i < kbestListAndGoldTreeDataStructures.size(); i++) {
            for(int g = 0; g < weights.length; g++) {
                gradients[g] = 2*lambda*weights[g] - differences[g];
            }
        }

        return gradients;
    }

    @Override
    public int dimension() {
        return initialWeights.length;
    }

    private static double score(KbestListAndGoldTreeDataStructure kbestListAndGoldTreeDataStructure, int index, double[] weights) {
        double num =0;
        double denom = 0;

        int[] kbestListFeatureVector = kbestListAndGoldTreeDataStructure.getKbestListFeatureVectors().get(index);

        for(int j = 0; j < kbestListFeatureVector.length; j++) {
            num+=weights[kbestListFeatureVector[j]];
        }
        num = Math.exp(num);

        for(int[] oneKbestListFeatureVector : kbestListAndGoldTreeDataStructure.getKbestListFeatureVectors()) {
            double temp = 0;
            for(int j = 0; j < oneKbestListFeatureVector.length; j++) {
                temp+=weights[oneKbestListFeatureVector[j]];
            }
            denom+=Math.exp(temp);
        }
        return (num/denom);
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
