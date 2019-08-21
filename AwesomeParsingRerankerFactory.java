package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.ParsingRerankerFactory;
import edu.berkeley.nlp.assignments.rerank.student.KbestListAndGoldTreeDataStructure;
import edu.berkeley.nlp.assignments.rerank.student.MyFeatureExtractor;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.StreamSupport;

public class AwesomeParsingRerankerFactory implements ParsingRerankerFactory {
    public AwesomeParsingReranker trainParserReranker(Iterable<Pair<KbestList, Tree<String>>> kbestListsAndGoldTrees) {
        AwesomeParsingReranker awesomeParsingReranker = new AwesomeParsingReranker();
        Indexer<String> featureIndexer = awesomeParsingReranker.getFeatureIndexer();

        System.out.println("Beginning feature extraction for " + StreamSupport.stream(kbestListsAndGoldTrees.spliterator(), false).count() + " training pair of kbestLists and gold trees.");

        Pair<ArrayList<KbestListAndGoldTreeDataStructure>, Indexer<String>> kbestListsAndGoldTreesDataStructureAndIndexer =
                MyFeatureExtractor.doFeatureExtraction(kbestListsAndGoldTrees, featureIndexer);

        ArrayList<KbestListAndGoldTreeDataStructure> kbestListAndGoldTreeDataStructures = kbestListsAndGoldTreesDataStructureAndIndexer.getFirst();
        featureIndexer = kbestListsAndGoldTreesDataStructureAndIndexer.getSecond();

        System.out.println("Finished extracting features for training pairs of kbestLists and gold trees. Proceeding to Perceptron learning.");

        //Perceptron

        double[] weights = new double[featureIndexer.size()];
//        double[] weights = new double[featureIndexer.size()+1];
//        Arrays.fill(weights, 0);
        for(int i = 0; i < weights.length; i++) {
            weights[i] = ThreadLocalRandom.current().nextDouble(-1, 1);
        }
        double[] averageIterationWeights = new double[weights.length];
//        double[] averageWeightsForOneKbestList = new double [weights.length];
        double[] averageWeightsForAllKbestLists = new double [weights.length];

        int iterations = 60;
        double learningRate = 1;

        for(int h = 0; h < iterations; h++) {
            int count = 0;

            Collections.shuffle(kbestListAndGoldTreeDataStructures);
//            Arrays.fill(averageWeightsForAllKbestLists, 0);
            //for each kbestList, pick the tree in the list with the best score; for each tree in that kbestList, sub that
            //tree's features and add the pseudo gold tree's features.
            for (KbestListAndGoldTreeDataStructure kbestListAndGoldTreeDataStructure : kbestListAndGoldTreeDataStructures) {
//                Arrays.fill(averageWeightsForOneKbestList, 0);
                ArrayList<int[]> kbestListFeatureVectors = kbestListAndGoldTreeDataStructure.getKbestListFeatureVectors();
                int argmaxIndex = -1;
                double bestPrediction = Double.NEGATIVE_INFINITY;
                for(int i = 0; i < kbestListFeatureVectors.size(); i++) {
                    int[] kbestListFeatureVector = kbestListFeatureVectors.get(i);
                    double prediction = 0;
                    for (int j = 0; j < kbestListFeatureVector.length; j++) {
                        prediction += weights[kbestListFeatureVector[j]];
                    }
//                    if(count == 0) {
//                        System.out.println("Prediction for tree " + i + " in this kbestlist is " + prediction);
//                    }
                    if (prediction > bestPrediction) {
                        bestPrediction = prediction;
                        argmaxIndex = i;
                    }
                }
                double lowestLoss = Double.POSITIVE_INFINITY;
                int pseudoGoldIndex = -1;
                ArrayList<Double> kbestListLosses = kbestListAndGoldTreeDataStructure.getKbestListLosses();
                for(int i = 0; i < kbestListLosses.size(); i++) {
                    double kbestListLoss = kbestListLosses.get(i);
//                    if(count == 0) {
//                        System.out.println("Loss for tree " + i + " in this kbestlist is " + kbestListLoss);
//                    }
                    if(kbestListLoss < lowestLoss) {
                        lowestLoss = kbestListLoss;
                        pseudoGoldIndex = i;
                    }
                }
                int[] pseudoGoldKbestListFeatureVector = kbestListFeatureVectors.get(pseudoGoldIndex);

//                int[] pseudoGoldKbestListFeatureVector = kbestListAndGoldTreeDataStructure.getGoldTreeFeatureVector();
//                for (int i = 0; i < kbestListFeatureVectors.size(); i++) {
//                    int[] kbestListFeatureVector = kbestListFeatureVectors.get(i);
//                    for (int j = 0; j < kbestListFeatureVector.length; j++) {
//                        weights[kbestListFeatureVector[j]] -= learningRate*weights[kbestListFeatureVector[j]];
//                    }
//                    for (int j = 0; j < pseudoGoldKbestListFeatureVector.length; j++) {
//                        weights[pseudoGoldKbestListFeatureVector[j]] += learningRate*weights[pseudoGoldKbestListFeatureVector[j]];
//                    }
//
//                    //bias
//                    weights[weights.length-1]-=learningRate*predictions[i];
//                    weights[weights.length-1]+=learningRate*predictions[argmaxIndex];
////
////                    for(int k = 0; k < averageWeightsForOneKbestList.length; k++) {
////                        averageWeightsForOneKbestList[k]+= weights[k];
////                    }
//                }
                if(argmaxIndex != -1) {
                    int[] argmaxKbestListFeatureVector = kbestListFeatureVectors.get(argmaxIndex);
                    for (int j = 0; j < argmaxKbestListFeatureVector.length; j++) {
                        weights[argmaxKbestListFeatureVector[j]] -= learningRate /* weights[argmaxKbestListFeatureVector[j]]*/;
                    }
                    for (int j = 0; j < pseudoGoldKbestListFeatureVector.length; j++) {
                        weights[pseudoGoldKbestListFeatureVector[j]] += learningRate /* weights[pseudoGoldKbestListFeatureVector[j]]*/;
                    }

                    for (int i = 0; i < averageWeightsForAllKbestLists.length; i++) {
                        averageWeightsForAllKbestLists[i] += weights[i];
                    }
                }
                else {
                    System.out.println("The argmaxIndex for kbestlist " + count + " in iteration " + h + " was -1");
                }

                count++;
                if (count % 1000 == 0) {
                    System.out.println("Iteration " + h + ": done learning on " + count + " training pairs of kbestLists and gold trees.");
                }
            }

            for(int i = 0; i < averageWeightsForAllKbestLists.length; i++) {
                averageIterationWeights[i] += averageWeightsForAllKbestLists[i]/(double)kbestListAndGoldTreeDataStructures.size();
            }
        }

        for(int i = 0; i < averageIterationWeights.length; i++) {
            averageIterationWeights[i] = averageIterationWeights[i]/(double)iterations;
        }

        int count = 0;
        for(int i = 0; i < averageIterationWeights.length; i++) {
          if(averageIterationWeights[i] != 0 ) {
            count++;
          }
        }
        System.out.println(count + " out of " + featureIndexer.size() + " weights are nonzero");


        awesomeParsingReranker.setFeatureIndexer(featureIndexer);
//        basicParsingReranker.setWeights(weights);
        awesomeParsingReranker.setWeights(averageIterationWeights);
        return awesomeParsingReranker;
    }
}

