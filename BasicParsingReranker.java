package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.ParsingReranker;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Indexer;

import java.util.List;

public class BasicParsingReranker implements ParsingReranker {
    private Indexer<String> featureIndexer = new Indexer<>();
    private double[] weights;

    Indexer<String> getFeatureIndexer() {
        return featureIndexer;
    }

    void setFeatureIndexer(Indexer<String> featureIndexer) {
        this.featureIndexer = featureIndexer;
    }

    public double[] getWeights() {
        return weights;
    }

    void setWeights(double[] weights) {
        this.weights = weights;
    }

    public Tree<String> getBestParse(List<String> sentence, KbestList kbestList) {
        MyFeatureExtractor myFeatureExtractor = new MyFeatureExtractor();
        //for each tree in kbestlist, convert to features and multiply with weights. Return the tree with the greatest product
        List<Tree<String>> kbestTrees = kbestList.getKbestTrees();
        int argmaxIndex = -1;
        double bestPrediction = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < kbestTrees.size(); i++) {
            int[] features = myFeatureExtractor.extractFeatures(kbestList, i, featureIndexer, false).getFirst();
            double prediction =0;
            for(int j = 0; j < features.length; j++) {
                prediction+=weights[features[j]];
            }
            if(prediction > bestPrediction) {
                bestPrediction = prediction;
                argmaxIndex = i;
            }
        }
        return kbestTrees.get(argmaxIndex);
    }
}
