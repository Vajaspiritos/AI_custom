using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI_Practice4
{
    public class Plan
    {
        // List to store layers with their number of neurons and activation function
        public List<(int numNeurons, Func<Node, Node> activation)> Layers { get; private set; }

        public Plan()
        {
            Layers = new List<(int numNeurons, Func<Node, Node> activation)>();
        }

        // Method to add a layer to the plan
        public void Add(int numNeurons, Func<Node, Node> activation)
        {
            Layers.Add((numNeurons, activation));
        }

        // Method to build the network based on the layers
        public List<Layer> Build(bool xavier = false)
        {
            List<Layer> network = new List<Layer>();

            // The previous layer is null initially since there is no input layer yet
            Layer previousLayer = null;

            foreach (var layerInfo in Layers)
            {
                // Create a new layer with the current layer's neurons and activation function
                Layer layer = new Layer(previousLayer?.Neurons.Count ?? 0, layerInfo.numNeurons, previousLayer, layerInfo.activation,xavier);

                // Add the new layer to the network
                network.Add(layer);

                // Update the previousLayer to be the current layer
                previousLayer = layer;
            }

            return network;
        }
    }

    

   
}

