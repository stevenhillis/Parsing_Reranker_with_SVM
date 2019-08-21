package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.SurfaceHeadFinder;
import edu.berkeley.nlp.ling.AnchoredTree;
import edu.berkeley.nlp.ling.Constituent;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.parser.EnglishPennTreebankParseEvaluator;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.Pair;

import java.util.*;

public class MyFeatureExtractor {
    int[] features;
    int featureCount;
    
    public static Pair<ArrayList<KbestListAndGoldTreeDataStructure>, Indexer<String>> doFeatureExtraction(Iterable<Pair<KbestList, Tree<String>>> kbestListsAndGoldTrees, Indexer<String> featureIndexer) {
        MyFeatureExtractor myFeatureExtractor = new MyFeatureExtractor();

        ArrayList<KbestListAndGoldTreeDataStructure> kbestListsAndGoldTreesDataStructure = new ArrayList<>();
        for(Pair<KbestList, Tree<String>> kbestListAndGoldTree : kbestListsAndGoldTrees) {
            KbestListAndGoldTreeDataStructure structure = new KbestListAndGoldTreeDataStructure();
            List<Tree<String>> kbestListTrees = kbestListAndGoldTree.getFirst().getKbestTrees();
            double[] kbestListTreesScores = kbestListAndGoldTree.getFirst().getScores();
            Tree<String> goldTree = kbestListAndGoldTree.getSecond();

            for(int i = 0; i < kbestListTrees.size(); i++) {
                Tree<String> kbestListTree = kbestListTrees.get(i);
                double kbestListTreeScore = kbestListTreesScores[i];
                Pair<int[], Indexer<String>> kbestListTreeFeaturesAndIndexer = myFeatureExtractor.extractFeatures(kbestListTree, kbestListTreeScore, featureIndexer, true);
                structure.addKbestListFeatureVector(kbestListTreeFeaturesAndIndexer.getFirst());
                structure.addKbestListLoss(computeLoss(kbestListTree, goldTree));
                featureIndexer = kbestListTreeFeaturesAndIndexer.getSecond();
            }
            Pair<int[], Indexer<String>> goldTreeFeaturesAndIndexer = myFeatureExtractor.extractFeatures(goldTree, 0, featureIndexer, true);
            structure.setGoldTreeFeatureVector(goldTreeFeaturesAndIndexer.getFirst());
            featureIndexer = goldTreeFeaturesAndIndexer.getSecond();
            kbestListsAndGoldTreesDataStructure.add(structure);
            if(kbestListsAndGoldTreesDataStructure.size()%1000==0) {
                System.out.println("Extracted features from " + kbestListsAndGoldTreesDataStructure.size() + " training pairs of kbestLists and gold trees. We have " + featureIndexer.size() + " features.");
                System.out.println("Gold tree feature vector had " + structure.getGoldTreeFeatureVector().length + " features.");
            }
        }
        System.out.println("Final feature count: " + featureIndexer.size());
        return new Pair(kbestListsAndGoldTreesDataStructure, featureIndexer);
    }

    public Pair<int[], Indexer<String>> extractFeatures(KbestList kbestList, int index, Indexer<String> featureIndexer, boolean addFeaturesToIndexer) {
        Tree<String> tree = kbestList.getKbestTrees().get(index);
        double score = kbestList.getScores()[index];
        return extractFeatures(tree, score, featureIndexer, addFeaturesToIndexer);
    }

    public Pair<int[], Indexer<String>> extractFeatures(Tree<String> tree, double score, Indexer<String> featureIndexer, boolean addFeaturesToIndexer) {
        List<String> words = tree.getYield();
        List<String> partsOfSpeech = tree.getPreTerminalYield();
        Collection<Constituent<String>> constituents = tree.toConstituentList();
        String punctuations = ".,:;?!-'*(){}[]";
        SurfaceHeadFinder shf = new SurfaceHeadFinder();

        features = new int[200];
        featureCount = 0;

        //length of sentence, split into 8 buckets
        int sentenceLength = words.size();
        String bucketLength;
        if(sentenceLength > 5 && sentenceLength <= 10) {
            bucketLength = "10";
        } else if (sentenceLength > 10 && sentenceLength <= 20) {
            bucketLength = "20";
        } else if (sentenceLength > 21) {
            bucketLength = "21+";
        } else {
            bucketLength = Integer.toString(sentenceLength);
        }

        //length
        featureIndexer = addFeature("Len="+bucketLength, featureIndexer, addFeaturesToIndexer);

        String bucketScore;
        if (score >= 0) {
            bucketScore = "0";
        } else if (score < 0 && score >= -5) {
            bucketScore = "-5";
        } else if (score < -5 && score >= -10) {
            bucketScore = "-10";
        } else if (score < -10 && score >= -15) {
            bucketScore = "-15";
        } else if (score < -15 && score >= -20) {
            bucketScore = "-20";
        } else if (score < -20 && score >= -25) {
            bucketScore = "-25";
        } else {
            bucketScore = "<-25";
        }

        //score
        featureIndexer = addFeature("Score="+bucketScore, featureIndexer, addFeaturesToIndexer);

        //bigram and POS neighbors
        for(int i = 0; i < sentenceLength-2; i++) {
            if(i!=0) {
                featureIndexer = addFeature("LeftNeighborPOS=" + partsOfSpeech.get(i-1), featureIndexer, addFeaturesToIndexer);
            }
            featureIndexer = addFeature("RightNeighborPOS=" + partsOfSpeech.get(i+1), featureIndexer, addFeaturesToIndexer);
            featureIndexer = addFeature("Bigram=" + words.get(i) + words.get(i+1), featureIndexer, addFeaturesToIndexer);
        }

        //split-aux, %, split-IN
        for(String word : words) {
            if (word.equals("am") || word.equals("was") || word.equals("are") || word.equals("is") || word.equals("were") || word.equals("been") || word.equals("being") ||
                    word.equals("Am") || word.equals("Was") || word.equals("Are") || word.equals("Is") || word.equals("Were") || word.equals("Been") || word.equals("Being")) {
                featureIndexer = addFeature(word + "~BE", featureIndexer, addFeaturesToIndexer);
            }
            if (word.equals("have") || word.equals("has") || word.equals("had") || word.equals("having") ||
                    word.equals("Have") || word.equals("Has") || word.equals("Had") || word.equals("Having")) {
                featureIndexer = addFeature(word + "~HAVE", featureIndexer, addFeaturesToIndexer);
            }
            if (word.equals("%")) {
                featureIndexer = addFeature("%", featureIndexer, addFeaturesToIndexer);
            }
            if(word.equals("while") || word.equals("While") || word.equals("as") || word.equals("As") || word.equals("if") || word.equals("If")) {
                featureIndexer = addFeature("SubConj", featureIndexer, addFeaturesToIndexer);
            }
            if(word.equals("that") || word.equals("That") || word.equals("for") || word.equals("For")) {
                featureIndexer = addFeature("Comp", featureIndexer, addFeaturesToIndexer);
            }
            if(word.equals("of") || word.equals("Of") || word.equals("in") || word.equals("In") || word.equals("from") || word.equals("From") || word.equals("about") || word.equals("About") || word.equals("with") || word.equals("With") || word.equals("up") || word.equals("Up")) {
                featureIndexer = addFeature("PP", featureIndexer, addFeaturesToIndexer);
            }
            if(word.equals("by") || word.equals("By")) {
                featureIndexer = addFeature("NM", featureIndexer, addFeaturesToIndexer);
            }
        }

        //rule; also Right-Rec-NP; also NP-B
        AnchoredTree<String> anchoredTree = AnchoredTree.fromTree(tree);
        for(AnchoredTree<String> subtree : anchoredTree.toSubTreeList()) {
            boolean allPreterminals = true;
            if(!subtree.isPreTerminal() && !subtree.isLeaf()) {
                String rule = "Rule=" + subtree.getLabel() + " ->";
                List<AnchoredTree<String>> subtreeChildren = subtree.getChildren();
                for(int i = 0; i < subtreeChildren.size(); i++) {
                    AnchoredTree<String> child = subtreeChildren.get(i);
                    String childLabel = child.getLabel();
                    rule += " " + childLabel;
                    if(!child.isPreTerminal()) {
                        allPreterminals = false;
                    }
                }
                if(subtreeChildren.get(subtreeChildren.size()-1).getLabel().equals("NP") && (subtree.getLabel().equals("NP"))) {
                    featureIndexer = addFeature("RightRecNP", featureIndexer, addFeaturesToIndexer);
                }

                if((subtree.getLabel().equals("NP")) && allPreterminals) {
                    featureIndexer = addFeature("NP-B", featureIndexer, addFeaturesToIndexer);
                }

                featureIndexer = addFeature(rule, featureIndexer, addFeaturesToIndexer);
            }
        }

        //first and last word of each span; length of span; also span shape; also parent label; also dominates-V, conjB, and conjA
        for(Constituent<String> constituent : constituents) {
            int firstInSpan = constituent.getStart();
            int lastInSpan = constituent.getEnd();
            String parentLabel = constituent.getLabel();
            List<String> preterminals = partsOfSpeech.subList(firstInSpan, lastInSpan);
            List<String> spanWords = words.subList(firstInSpan, lastInSpan);

            String head = preterminals.get(shf.findHead(parentLabel, preterminals));

            boolean dominatesV = false;
            boolean conjA = false;
            boolean conjB = false;
            boolean finiteVerb = false;

            featureIndexer = addFeature("ParentLabel=" + parentLabel, featureIndexer, addFeaturesToIndexer);
            featureIndexer = addFeature("FirstInSpan=" + words.get(firstInSpan), featureIndexer, addFeaturesToIndexer);
            featureIndexer = addFeature("LastInSpan=" + words.get(lastInSpan-1), featureIndexer, addFeaturesToIndexer);
            featureIndexer = addFeature("SpanLength=" + (lastInSpan - firstInSpan), featureIndexer, addFeaturesToIndexer);
            featureIndexer = addFeature("SpanHead=" + head, featureIndexer, addFeaturesToIndexer);

            if(firstInSpan != 0) {
                featureIndexer = addFeature("WordBeforeSpan=" + words.get(firstInSpan - 1), featureIndexer, addFeaturesToIndexer);
            }
            if(lastInSpan <= (words.size() -1)) {
                featureIndexer = addFeature("WordAfterSpan=" + words.get(lastInSpan), featureIndexer, addFeaturesToIndexer);
            }

            if(preterminals.size()==1) {
                if(preterminals.get(0).equals("DT")) {
                    featureIndexer = addFeature("Unary-DT", featureIndexer, addFeaturesToIndexer);
                } else if(preterminals.get(0).equals("RB")) {
                    featureIndexer = addFeature("Unary-RB", featureIndexer, addFeaturesToIndexer);
                }
            }

            for(String preterminal : preterminals) {
                if(preterminal.equals("MD") || (preterminal.toCharArray()[0] == 'V')) {
                    dominatesV = true;
                }
                if((preterminal.equals("VBD") || (preterminal.equals("VBP")) || (preterminal.equals("VBZ")))) {
                    finiteVerb = true;
                }
            }
            if(dominatesV) {
                featureIndexer = addFeature("DominatesVerbalNode", featureIndexer, addFeaturesToIndexer);
            }
//            if(finiteVerb) {
//                featureIndexer = addFeature("FiniteVerb", featureIndexer, addFeaturesToIndexer);
//            }

            if(parentLabel.equals("CC")) {
                for(String word : spanWords) {
                    if(word.equals("But") || word.equals("but")) {
                        conjB = true;
                    }
                    if(word.equals("&")) {
                        conjA = true;
                    }
                }
            }
            if(conjB) {
                featureIndexer = addFeature("ConjB", featureIndexer, addFeaturesToIndexer);
            }
            if(conjA) {
                featureIndexer = addFeature("ConjA", featureIndexer, addFeaturesToIndexer);
            }

            for(int i = firstInSpan; i < lastInSpan; i++) {
                String word = words.get(i);
                char c = word.toCharArray()[0];
                if(Character.isUpperCase(c)) {
                    featureIndexer = addFeature("Uppercase=" + i, featureIndexer, addFeaturesToIndexer);
                }
                if(Character.isLowerCase(c)) {
                    featureIndexer = addFeature("Lowercase=" + i, featureIndexer, addFeaturesToIndexer);
                }
                if(Character.isDigit(c)) {
                    featureIndexer = addFeature("Digit=" + i, featureIndexer, addFeaturesToIndexer);
                }
                if(punctuations.contains(Character.toString(c))) {
                    featureIndexer = addFeature("Punc=" + i, featureIndexer, addFeaturesToIndexer);
                }
            }

        }

        return new Pair(features, featureIndexer);
    }

    private Indexer<String> addFeature(String feature,Indexer<String> featureIndexer, boolean addNewFeatures) {
        if(addNewFeatures || featureIndexer.contains(feature)) {
            if(featureCount >= features.length) {
                features = Arrays.copyOf(features, 2*features.length);
            }
            features[featureCount] = (featureIndexer.addAndGetIndex(feature));
//            features[featureCount] = (featureIndexer.addAndGetIndex(feature));
            featureCount++;
        }
        return featureIndexer;
    }
    private static double computeLoss(Tree<String> kBestTree, Tree<String> goldTree) {
        EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String> eval = new EnglishPennTreebankParseEvaluator.LabeledConstituentEval(Collections.singleton("ROOT"), new HashSet<String>());
        double F1 = eval.evaluateF1(kBestTree, goldTree);
        return (1-F1);
    }
    
}
