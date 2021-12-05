package FFNN;

import java.util.ArrayList;
import java.util.Arrays;

import static FFNN.FileManagement.*;

public class Weights {
    public static void push_zeros_to_Weights(NeuralNetObjects neuralNetObjects)
    {
        int index, NumberOfWeights = 0;
        int topologySize = neuralNetObjects.topology.size();

        for (index = 0; index < topologySize - 1; index++)
        {
            NumberOfWeights += (neuralNetObjects.topology.get(index) + 1)*neuralNetObjects.topology.get(index + 1);
        }

        neuralNetObjects.weights = new ArrayList<>(Arrays.asList(new Float[NumberOfWeights]));
    }

    public static void push_zeros_to_Learning_table(NeuralNetObjects neuralNetObjects)
    {
        ArrayList<Float> InputRow = new ArrayList<>();
        ArrayList<Float> OutputRow = new ArrayList<>();
        int row, column;

        neuralNetObjects.learningInputs.clear();
        for (row = 0; row < neuralNetObjects.inputNodes; row++)
        {
            InputRow.add(0.0f);
        }
        for (column = 0; column < neuralNetObjects.patternCount; column++)
        {
            neuralNetObjects.learningInputs.add(InputRow);
        }

        neuralNetObjects.learningOutputs.clear();
        for (row = 0; row < neuralNetObjects.outputNodes; row++)
        {
            OutputRow.add(0.0f);
        }
        for (column = 0; column < neuralNetObjects.patternCount; column++)
        {
            neuralNetObjects.learningOutputs.add(OutputRow);
        }
    }

    public static void get_training_data_count(NeuralNetObjects neuralNetObjects)
    {
        ArrayList<String> fileContent = new ArrayList<>(readOrCreateFile(neuralNetObjects.trainingFilePath));

        if(fileContent.size()==0 || fileContent==null)
        {
            System.out.println("Cannot open " + neuralNetObjects.trainingFilePath);
            System.exit(-5);
        }

        int count = 0;

        for(int i = 0; i < fileContent.size(); i++)
        {
            for(int x = 0; x < fileContent.get(i).length(); x++)
            {
                if(fileContent.get(i).charAt(x) == '}')
                    count++;
            }
        }

        if(count % 2 == 0)
            neuralNetObjects.patternCount = count / 2;
        else
        {
            System.out.println("Training data error.");
            System.exit(-6);
        }

    }

    public static void loadTopology(NeuralNetObjects neuralNetObjects)
    {
        ArrayList<String> fileContent = new ArrayList<>(readOrCreateFile(neuralNetObjects.topologyFilePath));

        if(fileContent.size() == 0 || fileContent == null)
        {
            System.out.println("Cannot open " + neuralNetObjects.topologyFilePath);
            System.exit(-7);
        }

        for(int i = 0; i < fileContent.size(); i++)
        {
            String numberString = new String();
            for(int x = 0; x < fileContent.get(i).length(); x++)
            {
                char c = fileContent.get(i).charAt(x);
                if(c >= '0' && c <= '9')
                {
                    numberString += c;
                }
            }

            if(numberString != null && numberString.length() != 0)
            {
                neuralNetObjects.topology.add(Integer.parseInt(numberString));
                neuralNetObjects.inputNodes = neuralNetObjects.topology.get(0);
                neuralNetObjects.outputNodes = neuralNetObjects.topology.get(neuralNetObjects.topology.size() - 1);
                get_training_data_count(neuralNetObjects);
                push_zeros_to_Learning_table(neuralNetObjects);
                push_zeros_to_Weights(neuralNetObjects);
            }
        }
    }


    public static void load_training_data_from_file(NeuralNetObjects neuralNetObjects)
    {
        ArrayList<String> fileContent = new ArrayList<>(readOrCreateFile(neuralNetObjects.trainingFilePath));

        if(fileContent.size()==0 || fileContent==null)
        {
            System.out.println("Cannot open " + neuralNetObjects.trainingFilePath);
            System.exit(-8);
        }

        int trainingDataLine = 0;
        for(int fileLine = 0; fileLine < fileContent.size(); fileLine++)
        {
            if(fileContent.get(fileLine).contains("{"))
            {
                String[] bracketContent = fileContent.get(fileLine).split(",");
                ArrayList<Float> inputLine = new ArrayList<>();
                ArrayList<Float> outputLine = new ArrayList<>();
                int flag=0;
                for(int segment = 0; segment < bracketContent.length; segment++)
                {
                    String number = new String();
                    for(int i = 0; i < bracketContent[segment].length(); i++)
                    {
                        char c = bracketContent[segment].charAt(i);
                        if(flag==0 && c=='{')
                            flag=1;
                        else if(flag==1 && c=='{')
                            flag=2;
                        if(c == '+' || c == '-' || c == '.' || (c  >= '0' && c <= '9'))
                            number+=c;
                    }

                    if(number.length() != 0)
                    {
                        try
                        {
                            if(flag==1)
                                inputLine.add(Float.parseFloat(number));
                            else if(flag==2)
                                outputLine.add(Float.parseFloat(number));
                        }catch (Exception e)
                        {
                            System.out.println("Failed to parse number from this string:" + number);
                        }
                    }
                }
                System.out.println("Training =>> inputs:" + inputLine + " outputs: " + outputLine);
                neuralNetObjects.learningInputs.set(trainingDataLine, inputLine);
                neuralNetObjects.learningOutputs.set(trainingDataLine, outputLine);
                trainingDataLine++;
            }
        }

        System.out.println("learningInputs: " + neuralNetObjects.learningInputs);
        System.out.println("learningOutputs: " + neuralNetObjects.learningOutputs);
    }

    public static int get_number_of_weights_from_file(NeuralNetObjects neuralNetObjects)
    {
        int number_of_weights = 0;

        ArrayList<String> fileContent = new ArrayList<>(readOrCreateFile(neuralNetObjects.weightsFilePath));

        for (int i = 0; i < fileContent.size(); i++)
        {
            if(fileContent.get(i).length()!=0)
                number_of_weights++;
        }

        return number_of_weights;
    };
}
