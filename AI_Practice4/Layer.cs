using System;
using System.Collections.Generic;

namespace AI_Practice4
{
    public class Layer
    {
        public List<Node> Neurons; // List of neurons for this layer
        public List<Node> Biases; // Bias for each neuron
        public List<List<Node>> Weights; // 2D list for weights (previous layer x current layer)
        public Layer Previous; // Reference to the previous layer
        public Func<Node, Node> Activation; // Activation function
        public Dictionary<int,Func<Node, Node>> Exception = new Dictionary<int, Func<Node, Node>>();

        public virtual void feed(List<double> input) {

            for (int i = 0; i < input.Count; i++) {
                Neurons[i].VALUE = input[i];
            }
        
        }
        public Layer() { }
        public Layer(int numPreviousLayerNeurons, int numNeurons, Layer previous = null, Func<Node, Node> activation = null, bool useXavier = false)
        {
            Neurons = new List<Node>(numNeurons); // Initialize neurons
            Biases = new List<Node>(numNeurons); // Initialize biases
            Weights = new List<List<Node>>(numPreviousLayerNeurons); // Initialize weights

            Previous = previous;
            Activation = activation;

            // Initialize weights and biases
            InitializeNeurons(numNeurons);
            InitializeWeights(numPreviousLayerNeurons, numNeurons, useXavier);
            InitializeBiases(numNeurons);
        }

        public Layer( int numNeurons, bool useXavier = false)
        {
            Neurons = new List<Node>(numNeurons); // Initialize neurons
            Biases = new List<Node>(numNeurons); // Initialize biases
            Weights = new List<List<Node>>(0); // Initialize weights

            Previous = null;
            Activation = null;

            InitializeNeurons(numNeurons);
            // Initialize weights and biases
            InitializeWeights(0, numNeurons, useXavier);
            InitializeBiases(numNeurons);
        }

        private void InitializeNeurons(int numNeurons) { 
        
            for(int i=0;i<numNeurons;i++) Neurons.Add(new Node(0));
        
        }
        private void InitializeWeights(int numPreviousLayerNeurons, int numNeurons, bool useXavier)
        {
            Random rand = new Random();
            double limit = Math.Sqrt(6.0 / (numPreviousLayerNeurons + numNeurons)); // Xavier initialization range

            for (int i = 0; i < numPreviousLayerNeurons; i++)
            {
                var weightRow = new List<Node>(numNeurons);
                for (int j = 0; j < numNeurons; j++)
                {
                    double weightValue = useXavier
                        ? rand.NextDouble() * 2 * limit - limit // Xavier initialization
                        : Random.Shared.NextDouble()*2-1; // Default initialization to zero
                    weightRow.Add(new Node(weightValue));
                }
                Weights.Add(weightRow);
            }
        }

        private void InitializeBiases(int numNeurons)
        {
            for (int i = 0; i < numNeurons; i++)
            {
                Biases.Add(new Node(Random.Shared.NextDouble()*2-1)); // Initialize biases to zero
            }
        }

        public void Forward()
        {
            if (Previous == null)
            {
                throw new InvalidOperationException("Cannot perform forward pass without a previous layer.");
            }

            int numNeurons = Neurons.Count;
            int numPreviousNeurons = Previous.Neurons.Count;

            // Use multithreading to compute each neuron's output
            Parallel.For(0, numNeurons, g =>
            {
                Node weightedSum = 0d;
                int i = g;
                // Compute weighted sum of inputs
                for (int j = 0; j < numPreviousNeurons; j++)
                {
                    weightedSum += Previous.Neurons[j] * Weights[j][i];
                }

                // Add bias
                weightedSum += Biases[i];

                // Apply activation function
                if (!Exception.ContainsKey(i))
                {
                    Neurons[i] = Activation.Invoke(weightedSum);
                }
                else {
                    
                    Neurons[i] = Exception[i].Invoke(weightedSum);
                   // Console.WriteLine(Neurons.Count + "| triggered:" + Neurons[i].VALUE);
                }
                

            });
        }
        public override string ToString()
        {
            string ret = "";

            for (int i = 0; i < Neurons.Count; i++)
            {

                ret += "\nNeuron" + i+Environment.NewLine;
                ret += "    Neuron: "+Neurons[i].VALUE;
                ret += "    Bias: "+Biases[i].VALUE;
                
                for (int j = 0; j < Weights.Count; j++) {
                   ret += $"      Weigths {j}: "+string.Join("         \n", Weights[j][i].VALUE);

                }

            }
            return ret;
        }

    }


}

    

