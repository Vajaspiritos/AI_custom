using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;

namespace AI_Practice4
{
    public class SGD
    {
        private Network _network;
        private DataSet _dataset;
        private Func<List<Node>, List<Node>, Node> _lossFunction;

        // Constructor
        public SGD(Network network, DataSet dataset, Func<List<Node>, List<Node>, Node> lossFunction)
        {
            _network = network;
            _dataset = dataset;
            _lossFunction = lossFunction;
        }

        // Train method
        public void Train(int epochs, int batchSize=1)
        {

            for (int epoch = 0; epoch < epochs; epoch++)
            {
               // Console.WriteLine($"Epoch {epoch + 1}/{epochs}");

                // Shuffle dataset for each epoch (randomly permuting the data)
                _dataset.Shuffle();

                // Iterate over the dataset in mini-batches
                for (int i = 0; i < _dataset.Count; i ++)
                {
                    var inputs = _dataset.Inputs[i];   // Inputs
                    var targets = _dataset.Targets[i];  // Targets

                    // Perform forward pass for each input in the batch
                    List<Node> predictions = _network.forward(inputs);
                    

                    // Compute the loss (the last node in the prediction and targets list)
                    var lossNode = _lossFunction(predictions, targets.Select(X=>new Node(X)).ToList());

                    lossNode.Backpropagate();
                   if(i%batchSize==0) _network.ApplyGradientsAdam();

                }

                
            }
        }
    }
}
