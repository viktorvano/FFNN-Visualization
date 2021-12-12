package FFNN;

import com.sun.istack.internal.NotNull;

import java.awt.*;
import java.time.ZonedDateTime;
import java.util.ArrayList;

import static FFNN.FileManagement.readOrCreateFile;
import static FFNN.FileManagement.writeToFile;
import static FFNN.GeneralFunctions.showVectorValues;
import static FFNN.Weights.*;

public class NeuralNetwork {

    private TrainingThread trainingThread;
    private NeuralNetObjects neuralNetObjects;
    private ArrayList<Layer> m_layers; // m_layers[layerNum][neuronNum]

    private boolean netLoading;
    private boolean netTraining;
    private boolean stopTraining;

    private float m_error;
    private float m_recentAverageError;

    public NeuralNetwork(@NotNull NeuralNetObjects neuralNetObjects)
    {
        this.neuralNetObjects = neuralNetObjects;
        this.netLoading = true;
        this.netTraining = false;
        this.stopTraining = false;
        this.m_error = 0;
        this.m_recentAverageError = 0;
        int numLayers = neuralNetObjects.topology.size();
        System.out.println("Number of layers: " + numLayers);
        this.m_layers = new ArrayList<>();
        for (int layerNum = 0; layerNum < numLayers; layerNum++)
        {
            this.m_layers.add(new Layer());
            int numOutputs = layerNum == neuralNetObjects.topology.size() - 1 ? 0 : neuralNetObjects.topology.get(layerNum + 1);

            // We have made a new Layer, now fill it with neurons, and add a bias neuron to the layer.
            for (int neuronNum = 0; neuronNum <= neuralNetObjects.topology.get(layerNum); neuronNum++)
            {
                this.m_layers.get(this.m_layers.size()-1).add(new Neuron(this.neuralNetObjects, numOutputs, neuronNum));
                System.out.println("Made a neuron: " + neuronNum);
            }

            // Force the bias node's output value to 1.0. It's last neuron created above
            m_layers.get(m_layers.size()-1).peekLast().setOutputValue(1.0f);
        }
        this.loadNeuronWeights();
        if(getNumberOfWeightsFromFile(neuralNetObjects) == 0)
            trainNeuralNetwork();
        netLoading = false;
    }

    public boolean isNetLoading() {
        return netLoading;
    }
    public boolean isNetTraining(){
        return netTraining;
    }
    public void stopTraining()
    {
        stopTraining = true;
    }

    public void trainNeuralNetwork()
    {
        this.netTraining = true;
        this.trainingThread = new TrainingThread(this);
        this.trainingThread.start();
    }

    public void feedForward(ArrayList<Float> inputValues)
    {
        assert(inputValues.size() == m_layers.get(0).size() - 1);

        // Assign (latch) the input values into the input neurons
        for (int i = 0; i < inputValues.size(); i++)
        {
            m_layers.get(0).get(i).setOutputValue(inputValues.get(i));
        }

        // Forward propagate
        for (int layerNum = 1; layerNum < m_layers.size(); layerNum++)
        {
            Layer prevLayer = m_layers.get(layerNum - 1);
            for (int n = 0; n < m_layers.get(layerNum).size() - 1; n++)
            {
                m_layers.get(layerNum).get(n).feedForward(prevLayer);
            }
        }
    }
    public void backProp(ArrayList<Float> targetValues)
    {
        // Calculate overall net error (RMS of output neuron errors)
        Layer outputLayer = m_layers.get(m_layers.size()-1);
        m_error = 0.0f;

        for (int n = 0; n < outputLayer.size() - 1; n++)
        {
            float delta = targetValues.get(n) - outputLayer.get(n).getOutputValue();
            m_error += delta * delta;
        }
        m_error /= outputLayer.size() - 1; //get average errorsquared
        m_error = (float)Math.sqrt(m_error); // RMS

        // Implement a recent average measurement;

        m_recentAverageError = m_error;

        // Calculate output layer gradients
        for (int n = 0; n < outputLayer.size() - 1; n++)
        {
            outputLayer.get(n).calcOutputGradients(targetValues.get(n));
        }

        // Calculate gradients on hidden layers
        for (int layerNum = m_layers.size() - 2; layerNum > 0; layerNum--)
        {
            Layer hiddenLayer = m_layers.get(layerNum);
            Layer nextLayer = m_layers.get(layerNum + 1);

            for (int n = 0; n < hiddenLayer.size(); n++)
            {
                hiddenLayer.get(n).calcHiddenGradients(nextLayer);
            }
        }

        // For all layers from outputs to first hidden layer.
        // update connection weights

        for (int layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
        {
            Layer layer = m_layers.get(layerNum);
            Layer prevLayer = m_layers.get(layerNum - 1);

            for (int n = 0; n < layer.size() - 1; n++)
            {
                layer.get(n).updateInputWeights(prevLayer);
            }
        }
    }
    public void getResults(ArrayList<Float> resultValues)
    {
        resultValues.clear();

        for (int n = 0; n < m_layers.get(m_layers.size()-1).size() - 1; n++)
        {
            resultValues.add(m_layers.get(m_layers.size()-1).get(n).getOutputValue());
        }
    }
    public float getNeuronOutput(int x, int y)
    {
        return m_layers.get(x).get(y).getOutputValue();
    }
    public float getRecentAverageError() { return m_recentAverageError; }

    public void saveNeuronWeights()
    {
        neuralNetObjects.neuronIndex = 0;
        // Forward propagate
        for (int layerNum = 1; layerNum < m_layers.size(); layerNum++)
        {
            Layer prevLayer = m_layers.get(layerNum - 1);
            for (int n = 0; n < m_layers.get(layerNum).size() - 1; n++)
            {
                m_layers.get(layerNum).get(n).saveInputWeights(prevLayer);
            }
        }
        ZonedDateTime now = ZonedDateTime.now();
        writeToFile(neuralNetObjects.trainingStatusFilePath, now + "\nAverage Error: " + neuralNetObjects.averageError);
        Toolkit.getDefaultToolkit().beep();
    }

    public void loadNeuronWeights()
    {
        neuralNetObjects.neuronIndex = 0;

        //load weights from a file to Weights[]
        ArrayList<String> fileContent = new ArrayList<>(readOrCreateFile(neuralNetObjects.weightsFilePath));

        if(fileContent.size()!=0)
        {
            for (int index = 0; index < neuralNetObjects.weights.size(); index++)
            {
                if(fileContent.get(index).length()!=0)
                {
                    neuralNetObjects.weights.set(index, Float.parseFloat(fileContent.get(index)));
                }
            }
        }
        else
            System.out.println("File " + neuralNetObjects.weightsFilePath + " is empty.");


        // Forward propagate
        for (int layerNum = 1; layerNum < m_layers.size(); layerNum++)
        {
            Layer prevLayer = m_layers.get(layerNum - 1);
            for (int n = 0; n < m_layers.get(layerNum).size() - 1; n++)
            {
                m_layers.get(layerNum).get(n).loadInputWeights(prevLayer);
            }
        }
        Toolkit.getDefaultToolkit().beep();
    }

    private class TrainingThread extends Thread
    {
        NeuralNetwork myNet;
        NeuralNetObjects netObjects;

        private TrainingThread(NeuralNetwork net){
            this.myNet = net;
            this.netObjects = net.neuralNetObjects;
        }

        @Override
        public void run() {
            super.run();
            trainNeuralNet();
        }

        private void trainNeuralNet()
        {
            netObjects.input.clear();
            netObjects.target.clear();
            netObjects.result.clear();
            neuralNetObjects.trainingPass = 0;

            load_training_data_from_file(neuralNetObjects);

            System.out.println("Training started\n");
            this.netObjects.averageError = 1.0f;
            float currentTrainingError;
            float quickSaveErrorValue = 0.5f;
            boolean repeatTrainingCycle = false;
            while (true)
            {
                netObjects.trainingPass++;
                System.out.println("Pass: " + netObjects.trainingPass);

                //Get new input data and feed it forward:
                if(!repeatTrainingCycle)
                    netObjects.trainData.getNextInputs(netObjects);
                showVectorValues("Inputs:", netObjects.input);
                myNet.feedForward(netObjects.input);

                // Train the net what the outputs should have been:
                if(!repeatTrainingCycle)
                    netObjects.trainData.getTargetOutputs(netObjects);
                showVectorValues("Targets: ", netObjects.target);
                assert(netObjects.target.size() == netObjects.topology.get(netObjects.topology.size()-1));
                myNet.backProp(netObjects.target);//This function alters neurons

                // Collect the net's actual results:
                myNet.getResults(netObjects.result);
                showVectorValues("Outputs: ", netObjects.result);


                // Report how well the training is working, averaged over recent samples:
                System.out.println("Net recent average error: " + myNet.getRecentAverageError() + "\n\n");

                currentTrainingError = myNet.getRecentAverageError();
                this.netObjects.averageError = 0.99f*this.netObjects.averageError + 0.01f*currentTrainingError;
                System.out.println("Net average error: " + this.netObjects.averageError + "\n\n");
                repeatTrainingCycle = currentTrainingError > this.netObjects.averageError;

                if(this.netObjects.averageError < netObjects.trainingExitError
                  && netObjects.trainingPass > netObjects.minTrainingPasses)
                {
                    System.out.println("Exit due to low error :D\n\n");
                    myNet.saveNeuronWeights();
                    break;
                }if(this.netObjects.averageError < quickSaveErrorValue)
                {
                    quickSaveErrorValue = this.netObjects.averageError/2f;
                    myNet.saveNeuronWeights();
                }if(netObjects.trainingPass > netObjects.maxTrainingPasses)
                {
                    System.out.println("Training passes were exceeded...\n\n");
                    myNet.saveNeuronWeights();
                    break;
                }

                if(stopTraining)
                    break;
            }
            System.out.println("Training done.\n");
            stopTraining = false;
            this.myNet.netTraining = false;
            System.out.println("Neural Network loaded.");
        }

    }
}
