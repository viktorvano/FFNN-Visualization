package FFNN;

public class TrainingData {
    public static int getNextInputs(NeuralNetObjects neuralNetObjects)
    {
        neuralNetObjects.input.clear();

        if (neuralNetObjects.trainingLine >= neuralNetObjects.patternCount)
            neuralNetObjects.trainingLine = 0;

        for (int i = 0; i<neuralNetObjects.inputNodes; i++)
            neuralNetObjects.input.add(neuralNetObjects.learningInputs.get(neuralNetObjects.trainingLine).get(i));

        return neuralNetObjects.input.size();
    }

    public static int getTargetOutputs(NeuralNetObjects neuralNetObjects)
    {
        neuralNetObjects.target.clear();

        for (int i = 0; i<neuralNetObjects.outputNodes; i++)
            neuralNetObjects.target.add(neuralNetObjects.learningOutputs.get(neuralNetObjects.trainingLine).get(i));

        neuralNetObjects.trainingLine++;

        return neuralNetObjects.target.size();
    }
}
