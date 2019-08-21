package edu.berkeley.nlp.assignments.rerank.student;

import java.util.ArrayList;

public class KbestListAndGoldTreeDataStructure {
    private ArrayList<int[]> kbestListFeatureVectors = new ArrayList<>();
    private int[] goldTreeFeatureVector;
    private ArrayList<Double> kbestListLosses = new ArrayList<>();
    private int bestAvailableTreeIndex;

    public ArrayList<int[]> getKbestListFeatureVectors() {
        return kbestListFeatureVectors;
    }

    public void setKbestListFeatureVectors(ArrayList<int[]> kbestListFeatureVectors) {
        this.kbestListFeatureVectors = kbestListFeatureVectors;
    }

    public int[] getGoldTreeFeatureVector() {
        return goldTreeFeatureVector;
    }

    public void setGoldTreeFeatureVector(int[] goldTreeFeatureVector) {
        this.goldTreeFeatureVector = goldTreeFeatureVector;
    }

    public ArrayList<Double> getKbestListLosses() {
        return kbestListLosses;
    }

    public void setKbestListLosses(ArrayList<Double> kbestListLosses) {
        this.kbestListLosses = kbestListLosses;
    }

    public int getBestAvailableTreeIndex() {
        return bestAvailableTreeIndex;
    }

    public void setBestAvailableTreeIndex(int bestAvailableTreeIndex) {
        this.bestAvailableTreeIndex = bestAvailableTreeIndex;
    }

    public void addKbestListFeatureVector(int[] kbestListFeatureVector) {
        kbestListFeatureVectors.add(kbestListFeatureVector);
    }

    public void addKbestListLoss(double kbestListLoss) {
        kbestListLosses.add(kbestListLoss);
    }
}
