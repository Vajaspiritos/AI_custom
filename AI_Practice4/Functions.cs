using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI_Practice4
{
    public static class Functions
    {
       
            public static Node Tanh(Node a)
            {
                double value = Math.Tanh(a.VALUE);
                double grad = 1 - value * value; // Derivative of tanh(x) is 1 - tanh(x)^2
                return new Node(value, new List<Node> { a }, new List<double> { grad });
            }

            public static Node Sigmoid(Node a)
            {
                double value = 1 / (1 + Math.Exp(-a.VALUE));
                double grad = value * (1 - value); // Derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
                return new Node(value, new List<Node> { a }, new List<double> { grad });
            }

            public static Node ReLU(Node a)
            {
                double value = Math.Max(0, a.VALUE);
                double grad = a.VALUE > 0 ? 1.0 : 0.0; // Derivative of ReLU is 1 if x > 0, else 0
                return new Node(value, new List<Node> { a }, new List<double> { grad });
            }

        public static Node LeakyReLU(Node a)
        {
            double alpha = 0.01;
            double value = a.VALUE > 0 ? a.VALUE : alpha * a.VALUE;
            double grad = a.VALUE > 0 ? 1.0 : alpha; // Derivative is 1 for positive values, and alpha for negative values
            return new Node(value, new List<Node> { a }, new List<double> { grad });
        }

        public static Node Softplus(Node a)
        {
            double value = Math.Log(1 + Math.Exp(a.VALUE));
            double grad = 1 / (1 + Math.Exp(-a.VALUE)); // Derivative of softplus is sigmoid(x)
            return new Node(value, new List<Node> { a }, new List<double> { grad });
        }

        public static Node PDF(Node a, Node mu, Node sigma)
        {
            //if (sigma.V == 0) sigma.V = 1e-5f;
            // return ((Node)1f/(sigma*(Node)MathF.Sqrt(2f*MathF.PI)))*Node.Exp((Node)(-1f/2f)* ((a - mu) / sigma)* ((a - mu) / sigma));
            //return -1f / 2 * ((a - mu) / (sigma)) * ((a - mu) / (sigma)) - Node.Log(sigma * MathF.Sqrt(MathF.PI * 2));
            return (-1d/2d) * Math.Log(2*Math.PI) - Node.Log(sigma) - ( Node.Pow(a-mu,2)/(2*Node.Pow(sigma,2))   );
        }
        public static List<Node> Softmax(Node[] x)
        {
            // Calculate exp(x) for each element in the input
            Node[] expValues = new Node[x.Length];
            Node sumExp = 0f;

            for (int i = 0; i < x.Length; i++)
            {
                expValues[i] = Node.Exp(x[i]);
                sumExp += expValues[i];
            }

            // Normalize each exp(x) by dividing by the sum of all exp(x)
            List<Node> softmaxValues = new List<Node>();
            for (int i = 0; i < x.Length; i++)
            {
                softmaxValues.Add( expValues[i] / sumExp);
            }

            return softmaxValues;
        }

     
        public static Node LogProbs(Node a)
        {
            if (a.VALUE <= 0 || a.VALUE > 1)
                throw new ArgumentException("Log probability is undefined for values outside the range [0, 1].");

            double value = Math.Log(a.VALUE); // Log of the probability
            double grad = 1 / a.VALUE;        // Derivative of log(x) is 1/x
            return new Node(value, new List<Node> { a }, new List<double> { grad });
        }

        private static readonly Random RandomGenerator = new Random();
        public static double Bellman(double mean, double stdDev)
        {
            if (stdDev <= 0)
                throw new ArgumentException("Standard deviation must be positive", nameof(stdDev));

            // Generate two uniformly distributed random numbers in the range (0, 1)
            double u1 = 1.0 - RandomGenerator.NextDouble(); // Avoid log(0)
            double u2 = RandomGenerator.NextDouble();

            // Apply the Box-Muller transform
            double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);

            // Scale and shift to match the desired mean and standard deviation
            return mean + (double)(z0 * stdDev);
        }

        public static List<Node> NormalizeAdvantages(List<Node> advantages)
        {
            // Calculate mean and standard deviation of the advantages
            double mean = advantages.Average(a => a.VALUE);  // assuming Node has a Value property
            double std = Math.Sqrt(advantages.Average(a => Math.Pow(a.VALUE - mean, 2)));

            // Normalize the advantages
            List<Node> normalizedAdvantages = new List<Node>();
            foreach (var advantage in advantages)
            {
                double normalizedValue = (advantage.VALUE - mean) / (std + 1e-8);  // Adding small epsilon to avoid division by zero
                normalizedAdvantages.Add(new Node(normalizedValue));
            }

            return normalizedAdvantages;
        }

        public static (List<Node> advantages, List<double> returns) GAE(List<Step> steps, double gamma = 0.99, double lambda = 0.95)
        {
            int n = steps.Count;
            List<Node> advantages = new List<Node>(new Node[n]);
            List<Node> returns = new List<Node>(new Node[n]);

            Node lastGAE = new Node(0);  // Initialize to 0 for the last GAE
            Node lastValue = new Node(0); // Final value estimate for the terminal state

            // Compute GAE and returns backwards from the end of the trajectory
            for (int t = n - 1; t >= 0; t--)
            {
                // Get the value estimate from the critic
                Node value = steps[t].CriticEstimate;

                // Bellman backup for GAE
                Node delta = steps[t].reward + gamma * lastValue * (1 - (steps[t].done ? 1 : 0)) - value;
                advantages[t] = lastGAE = delta + gamma * lambda * (1 - (steps[t].done ? 1 : 0)) * lastGAE;

                // Compute returns: GAE + value estimate
                returns[t] = advantages[t] + value;

                // Update last value for the next step
                lastValue = steps[t].reward + gamma * lastValue * (1 - (steps[t].done ? 1 : 0));
            }

            // Return both advantages and returns as a tuple
            return (advantages, returns.Select(X=>X.VALUE).ToList());
        }


        public static Node MeanSquaredError(List<Node> predictions, List<Node> targets)
        {
            Node sum = 0;
            for (int i = 0; i < predictions.Count; i++)
            {

                sum += Node.Pow(predictions[i] - targets[i],2);
                //sum += (targets[i] - predictions[i]) * (targets[i] - predictions[i]);
            }

            return sum / predictions.Count;
        }

        public static Node MeanAbsoluteError(List<Node> predictions, List<Node> targets)
        {
            return null;
        }


        public static Node Linear(Node a)
        {
            double value = a.VALUE; // Output is the same as input
            double grad = 1.0;      // Derivative of linear function is 1
            return new Node(value, new List<Node> { a }, new List<double> { grad });
        }

        public static Node Step(Node a)
        {
            // Thresholding function: output is -1, 0, or 1 based on value
            double value = a.VALUE > 0.5 ? a.VALUE : (a.VALUE < -0.5 ? a.VALUE : 0.0);

            // Gradient is 0 everywhere except at the thresholds (-0.5 and 0.5),
            // but we approximate it for autograd as a small gradient around the decision boundaries.
            double grad = (a.VALUE > -0.5 && a.VALUE < 0.5) ? 1.0 : 0.0;

            // Return a new Node with the computed value and gradient
            return new Node(value, new List<Node> { a }, new List<double> { grad });
        }







    }

}

