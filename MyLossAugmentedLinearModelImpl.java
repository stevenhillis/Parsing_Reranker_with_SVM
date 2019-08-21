package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.LossAugmentedLinearModel;
import edu.berkeley.nlp.util.IntCounter;

import java.util.ArrayList;
import java.util.Arrays;

public class MyLossAugmentedLinearModelImpl implements LossAugmentedLinearModel<KbestListAndGoldTreeDataStructure> {
    private int timesCalled = 0;

    @Override
    public UpdateBundle getLossAugmentedUpdateBundle(KbestListAndGoldTreeDataStructure datum, IntCounter weights) {
        timesCalled++;
        if(timesCalled % 1000 == 0) {
            System.out.println("Done learning on " + timesCalled + " training pairs of kbestLists and gold trees.");
        } else if (timesCalled == 36765) {
            timesCalled = 0;
        }

        int[] goldTreeFeatures = datum.getGoldTreeFeatureVector();
        IntCounter goldFeatures = new IntCounter(goldTreeFeatures.length);
        for(int i = 0; i < goldTreeFeatures.length; i++) {
            goldFeatures.put(goldTreeFeatures[i], 1);
        }

        ArrayList<int[]> kbestListFeatureVectors = datum.getKbestListFeatureVectors();
        ArrayList<Double> kbestListLosses = datum.getKbestListLosses();
        int argmaxIndex = -1;
        double bestPrediction = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < kbestListFeatureVectors.size(); i++) {
            int[] kbestListFeatureVector = kbestListFeatureVectors.get(i);
            double prediction = 0;
            int size = kbestListFeatureVector.length;
            for(int j = 0; j < size; j++) {
                prediction+=weights.get(kbestListFeatureVector[j]);
            }
            prediction += kbestListLosses.get(i);

            if(prediction > bestPrediction) {
                bestPrediction = prediction;
                argmaxIndex = i;
            }
        }

        int[] argmaxFeatureVector = kbestListFeatureVectors.get(argmaxIndex);
        IntCounter lossAugGuessFeatures = new IntCounter(argmaxFeatureVector.length);
        for(int i = 0; i < argmaxFeatureVector.length; i++) {
            lossAugGuessFeatures.put(argmaxFeatureVector[i], 1);
        }

        double argmaxKbestListLoss = kbestListLosses.get(argmaxIndex);

        return new UpdateBundle(goldFeatures, lossAugGuessFeatures, argmaxKbestListLoss);
    }
}
