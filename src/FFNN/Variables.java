package FFNN;

import java.util.ArrayList;

public class Variables {
    /*
        vector (c++)   <===========> linkedlist(java)
        v.front()      <===========> l.peekFirst()
        v.back()       <===========> l.peekLast()
        v.push_back(x) <===========> l.add(x)
        v.pop_back()   <===========> l.pollLast()
    */
    public static int patternCount;
    public static int inputNodes;
    public static int outputNodes;
    public static double velocity = 0.1; // overall net learning rate [0.0..1.0]
    public static double momentum = 0.5; // momentum multiplier of last deltaWeight [0.0..n]
    public static double definedRecentAverageSmoothingFactor = 0.0;


    public static ArrayList<Integer> topology;
    public static final ArrayList<ArrayList<Double>> learningInputs = new ArrayList<>();
    public static final ArrayList<ArrayList<Double>> learningOutputs = new ArrayList<>();
    public static ArrayList<Double> weights = new ArrayList<>();
    public static int neuronIndex = 0;
    public static int trainingLine = 0;// Has to be initialized 0
    public static ArrayList<Double> input, target, result;
    public static int trainingPass = 0;
}
