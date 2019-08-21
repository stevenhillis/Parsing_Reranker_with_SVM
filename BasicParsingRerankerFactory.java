package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.ParsingReranker;
import edu.berkeley.nlp.assignments.rerank.ParsingRerankerFactory;
import edu.berkeley.nlp.assignments.rerank.PrimalSubgradientSVMLearner;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.parser.EnglishPennTreebankParseEvaluator;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.IntCounter;
import edu.berkeley.nlp.util.Pair;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.StreamSupport;

public class BasicParsingRerankerFactory implements ParsingRerankerFactory {

    public BasicParsingReranker trainParserReranker(Iterable<Pair<KbestList,Tree<String>>> kbestListsAndGoldTrees) {
        BasicParsingReranker basicParsingReranker = new BasicParsingReranker();
        Indexer<String> featureIndexer = basicParsingReranker.getFeatureIndexer();

        System.out.println("Beginning feature extraction for " + StreamSupport.stream(kbestListsAndGoldTrees.spliterator(), false).count() + " training pair of kbestLists and gold trees.");

        Pair<ArrayList<KbestListAndGoldTreeDataStructure>, Indexer<String>> kbestListsAndGoldTreesDataStructureAndIndexer =
                MyFeatureExtractor.doFeatureExtraction(kbestListsAndGoldTrees, featureIndexer);

        ArrayList<KbestListAndGoldTreeDataStructure> kbestListsAndGoldTreesDataStructure = kbestListsAndGoldTreesDataStructureAndIndexer.getFirst();
        featureIndexer = kbestListsAndGoldTreesDataStructureAndIndexer.getSecond();

        System.out.println("Finished extracting features for training pairs of kbestLists and gold trees. Proceeding to SVM learning.");

        int numFeatures = featureIndexer.size();
        //L2 norm weight
        double regConstant = 1e-3;
        double stepSize = 1e-10;
        int batchSize = 100;
        int epochs = 70;

//        double[] weights = new double[featureIndexer.size()];
//        Arrays.fill(weights, 0);
//
//        IntCounter wrappedWeights = IntCounter.wrapArray(weights, weights.length);
        IntCounter wrappedWeights = new IntCounter(featureIndexer.size());
        MyLossAugmentedLinearModelImpl myLossAugmentedLinearModel = new MyLossAugmentedLinearModelImpl();
      PrimalSubgradientSVMLearner<KbestListAndGoldTreeDataStructure> primalSubgradientSVMLearner = new PrimalSubgradientSVMLearner<>(stepSize, regConstant, numFeatures, batchSize);
        IntCounter primalSubgradientSVMResult = primalSubgradientSVMLearner.train(wrappedWeights, myLossAugmentedLinearModel, kbestListsAndGoldTreesDataStructure, epochs);

//        double[] learnedWeights = primalSubgradientSVMResult.toArray(featureIndexer.size());
        double[] learnedWeights = new double[featureIndexer.size()];
        for(int i = 0; i < learnedWeights.length; i++) {
            learnedWeights[i] = primalSubgradientSVMResult.get(i);
        }

        int count = 0;
        for(int i = 0; i < learnedWeights.length; i++) {
            if(learnedWeights[i]!=0) {
                count++;
            }
        }
        System.out.println(count + " out of " + featureIndexer.size() + " weights are nonzero");

        System.out.println("Finished learning weights.");

        basicParsingReranker.setFeatureIndexer(featureIndexer);
        basicParsingReranker.setWeights(learnedWeights);
        return basicParsingReranker;
    }
}