package FFNN;

import java.util.ArrayList;
import static FFNN.FileManagement.*;
import static FFNN.GeneralFunctions.*;

public class Neuron {
    NeuralNetObjects neuralNetObjects;
    private float eta; // [0.0..1.0] overall network training rate
    private float alpha; // [0.0..n] multiplier of last weight change (momentum)

    public Neuron(NeuralNetObjects neuralNetObjects, int numOutputs, int myIndex)
    {
        this.neuralNetObjects = neuralNetObjects;
        this.eta = this.neuralNetObjects.velocity;
        this.alpha = this.neuralNetObjects.momentum;
        m_outputWeights = new ArrayList<>();
        m_outputWeights.clear();

        for (int c = 0; c < numOutputs; c++)
        {
            m_outputWeights.add(new Connection());
        }

        m_myIndex = myIndex;
    }

    public void setOutputValue(float value) { m_outputValue = value; }
    public float getOutputValue() { return m_outputValue; }
    public void feedForward(Layer prevLayer)
    {
        float sum = 0.0f;

        // Sum the previous layer's outputs (which are inputs)
        // Include the bias node from the previous layer.

        for (int n = 0; n < prevLayer.size(); n++)
        {
            sum += prevLayer.get(n).getOutputValue() * prevLayer.get(n).m_outputWeights.get(m_myIndex).weight;
        }

        m_outputValue = Neuron.transferFunction(sum);
    }

    public void calcOutputGradients(float targetValue)
    {
        float delta = targetValue - m_outputValue;
        m_gradient = delta * transferFunctionDerivative(m_outputValue);
    }

    public void calcHiddenGradients(Layer nextLayer)
    {
        float dow = sumDOW(nextLayer);
        m_gradient = dow * transferFunctionDerivative(m_outputValue);
    }

    public void updateInputWeights(Layer prevLayer)
    {
        // The weights to updated are in the Connection container
        // in the neurons in the preceding layer
        for (int n = 0; n < prevLayer.size(); n++)
        {
            Neuron neuron = prevLayer.get(n);
            float oldDeltaWeight = neuron.m_outputWeights.get(m_myIndex).deltaWeight;

            float newDeltaWeight =
                    // Individual input, magnified by the gradient and train rate:
                    eta // 0.0==slowlearner; 0.2==medium learner; 1.0==reckless learner
                            * neuron.getOutputValue()
                            * m_gradient
                            // Also add momentum = a fraction of the previous delta weight
                            + alpha // 0.0==no momentum; 0.5==moderate momentum
                            * oldDeltaWeight;
            neuron.m_outputWeights.get(m_myIndex).deltaWeight = newDeltaWeight;
            neuron.m_outputWeights.get(m_myIndex).weight += newDeltaWeight;
        }
    }

    public void saveInputWeights(Layer prevLayer)
    {
        // The weights to updated are in the Connection container
        // in the neurons in the preceding layer

        for (int n = 0; n < prevLayer.size(); n++)
        {
            Neuron neuron = prevLayer.get(n);
            neuralNetObjects.weights.set(neuralNetObjects.neuronIndex, neuron.m_outputWeights.get(m_myIndex).weight);
            neuralNetObjects.neuronIndex++;
        }

        if (neuralNetObjects.neuronIndex == neuralNetObjects.weights.size())
        {
            //save weights from Weights[] to a file
            String strWeights = new String();

            for (int index = 0; index < neuralNetObjects.weights.size(); index++)
            {
                strWeights += (formatFloatToString12(neuralNetObjects.weights.get(index)) + "\n");
            }
            writeToFile(neuralNetObjects.weightsFilePath, strWeights);
        }
    }

    public void loadInputWeights(Layer prevLayer)
    {
        for (int n = 0; n < prevLayer.size(); n++)
        {
            Neuron neuron = prevLayer.get(n);
            if(neuralNetObjects.weights.get(neuralNetObjects.neuronIndex) != null)
                neuron.m_outputWeights.get(m_myIndex).weight = neuralNetObjects.weights.get(neuralNetObjects.neuronIndex);
            this.neuralNetObjects.neuronIndex++;
        }
    }

    private float sumDOW(Layer nextLayer)
    {
        float sum = 0.0f;

        // Sum our contributions of the errors at the nodes we feed
        for (int n = 0; n < nextLayer.size() - 1; n++)
        {
            sum += m_outputWeights.get(n).weight * nextLayer.get(n).m_gradient;
        }

        return sum;
    }

    private static float transferFunction(float x)
    {
        // tanh - output range [-1.0..1.0]
        return (float)Math.tanh(x);
    }

    private float transferFunctionDerivative(float x)
    {
        // tanh derivative
        return (float) (1.0f - (float)Math.pow(Math.tanh(x), 2.0));// approximation return 1.0 - x*x;
    }

    private float m_outputValue;
    private ArrayList<Connection> m_outputWeights;
    private int m_myIndex;
    private float m_gradient;
}
