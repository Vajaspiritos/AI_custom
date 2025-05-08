using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
namespace AI_Practice4
{
    public class DataSet
    {
        public List<List<double>> Inputs { get; private set; }
        public List<List<double>> Targets { get; private set; }

        public DataSet()
        {
            Inputs = new List<List<double>>();
            Targets = new List<List<double>>();
        }

        // Add input-output pair to the dataset
        public void Add(List<double> input, List<double> target)
        {
            if (input == null || target == null)
                throw new ArgumentNullException("Input and target cannot be null");

            // You can also add input validation here if necessary
            Inputs.Add(input);
            Targets.Add(target);
        }

        // Return a specific input and target pair
        public (List<double> input, List<double> target) GetSample(int index)
        {
            if (index < 0 || index >= Inputs.Count)
                throw new ArgumentOutOfRangeException(nameof(index), "Index is out of range");

            return (Inputs[index], Targets[index]);
        }

        // Get the total number of samples in the dataset
        public int Count => Inputs.Count;

        // Shuffle dataset (optional, for training purposes)
        public void Shuffle()
        {
            Random rng = new Random();
            int n = Inputs.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                var value = Inputs[k];
                Inputs[k] = Inputs[n];
                Inputs[n] = value;

                var target = Targets[k];
                Targets[k] = Targets[n];
                Targets[n] = target;
            }
        }
    }
}
