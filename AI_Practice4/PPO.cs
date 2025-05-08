using System;
using System.Collections.Generic;
using System.Linq;

namespace AI_Practice4
{
    public class PPO
    {
        private Network policyNetwork; // Main policy network.
        private Network valueNetwork;  // Value network for estimating returns.
        private IEnviroment env;       // Environment interface.
        private double gamma = 0.95;   // Discount factor.
        private double lambda = 0.95;//0.95;  // GAE lambda for smoothing advantages.
        private double clipEpsilon = 0.2; // Clipping parameter for PPO.
        private int batchSize = 128;//128;    // Batch size for training.
        private int epochs = 10;//10;       // Number of epochs per update.
        private int bufferCapacity = 2048;//2048; // Rollout buffer capacity.
        private double bestReward = 0;
        private List<Step> rolloutBuffer = new List<Step>();

        public PPO(Network policyNetwork, Network valueNetwork, IEnviroment env)
        {
            this.policyNetwork = policyNetwork;
            this.valueNetwork = valueNetwork;
            this.env = env;
        }
        public double[] forward(List<double> inputs) {

            var output = policyNetwork.forward(inputs);
            var mus = output.Take(output.Count / 2).ToList();  // First half: means.
            var sigmas = output.Skip(output.Count / 2).Select(Node.Exp).ToList(); // Second half: log(sigmas).
            Random rnd = new Random();
            double[] action = mus.Select((mu, idx) =>
            {
                return Functions.Bellman(mu.VALUE, sigmas[idx].VALUE);
            }).ToArray();
            return action;
        }
        public void Learn(int episodes)
        {
            for (int i = 0; i < episodes; i++)
            {
                env.Reset();
                Console.Clear();
                Console.WriteLine($"{Math.Floor((double)i / (double)episodes * 10000) / 100}% Done.");

                // Collect rollouts
                CollectRollouts();
                Console.WriteLine("Rolled out");
                
                // Train networks using the collected rollouts
                TrainPolicyAndValue();
            }
        }

        private void CollectRollouts()
        {
            rolloutBuffer.Clear();
            int steps = 0;

            while (steps < bufferCapacity)
            {
                Step step = new Step();
                step.State = env.GetState().ToList();
                
                // Get action distribution from policy network
                var output = policyNetwork.forward(step.State);
                var mus = output.Take(output.Count / 2).ToList();  // First half: means.
                var sigmas = output.Skip(output.Count / 2).Select(Node.Exp).ToList(); // Second half: log(sigmas).



                // Sample action from Gaussian distribution
                Random rnd = new Random();
                List<double> action = mus.Select((mu, idx) =>
                {
                    return Functions.Bellman(mu.VALUE, sigmas[idx].VALUE);
                }).ToList();

                // Execute action in the environment
                (double[] nextState, double reward, bool done) = env.Step(action.ToArray());
                step.Actions = action.ToArray();
                step.reward = reward;
                step.Next_State = nextState.ToList();
                step.done = done;
                step.Logprobs = LogGaussian(step.Actions.ToList(), mus, sigmas).VALUE;

                // Store advantage and value for GAE later
                step.CriticEstimate = valueNetwork.forward(step.State)[0]; // State value estimate.
                rolloutBuffer.Add(step);

                steps++;
                if (done) env.Reset();
            }

            // Compute advantages and returns for the buffer
            ComputeAdvantagesAndReturns();
        }

        private void ComputeAdvantagesAndReturns()
        {
            Node lastValue = 0;
            Node lastAdvantage = 0;

            for (int i = rolloutBuffer.Count - 1; i >= 0; i--)
            {
                var step = rolloutBuffer[i];
                Node tdError = step.reward + gamma * lastValue * (step.done ? 0 : 1) - step.CriticEstimate;
                step.Advantage = (tdError + gamma * lambda * lastAdvantage * (step.done ? 0 : 1)).VALUE;
                step.Return = (step.reward + gamma * lastValue * (step.done ? 0 : 1)).VALUE;

                lastValue = step.CriticEstimate;
                lastAdvantage = step.Advantage;
            }


            double meanAdvantage = rolloutBuffer.Average(s => s.Advantage);
            double stdAdvantage = Math.Sqrt(rolloutBuffer.Average(s => Math.Pow(s.Advantage - meanAdvantage, 2)));
            foreach (var step in rolloutBuffer)
                step.Advantage = (step.Advantage - meanAdvantage) / (stdAdvantage + 1e-8);

        }

        private void TrainPolicyAndValue()
        {
            Random rnd = new Random();
             var shuffled = rolloutBuffer.OrderBy(x => rnd.Next());
            //var shuffled = rolloutBuffer;
            int nMini = (shuffled.Count() / batchSize);
            for (int e = 0; e < epochs; e++)
            {
                Console.WriteLine($"Epoch {e+1} / {epochs} ");

                //var minibatch = rolloutBuffer.OrderBy(x => rnd.Next()).Take(batchSize).ToList();
                var minibatch = shuffled.Skip(e * batchSize).Take(batchSize);


                double totalloss = 0;
                double totalreward = 0;
                double entropyCoeff = 0.01;
                foreach (var step in minibatch)
                {

                   // Console.WriteLine("0");

                    // Compute the new log probability
                    var newOutput = policyNetwork.forward(step.State);
                    var newMus = newOutput.Take(newOutput.Count / 2).ToList();
                    var newSigmas = newOutput.Skip(newOutput.Count / 2).Select(Node.Exp).ToList();
                    Node newLogProb = LogGaussian(step.Actions.ToList(), newMus, newSigmas);
                   // Console.WriteLine("1");
                    // Compute ratio and clip
                    Node ratio = Node.Exp(newLogProb - step.Logprobs);
                    Node clippedRatio = Node.Clamp(ratio, 1 - clipEpsilon, 1 + clipEpsilon);
                    Node policyLoss = -1*Node.Min(ratio * step.Advantage, clippedRatio * step.Advantage);


                   // Node entropy = ComputeEntropy(newSigmas);
                 //   policyLoss = policyLoss - entropyCoeff * entropy;
                    // Backpropagate policy loss
                    policyLoss.Backpropagate();
                    
                 //   Console.WriteLine("3");

                    // Value update
                    Node valueLoss = Node.Pow(step.Return - valueNetwork.forward(step.State)[0], 2);
                    valueLoss.Backpropagate();
                    totalloss += valueLoss.VALUE;
                    totalreward += step.reward;
                   // Console.WriteLine("4");

                }

                if (totalreward > bestReward)
                {
                    bestReward = totalreward;
                    policyNetwork.save(false, false, "Best.js");
                }

                policyNetwork.loss.Add(totalloss/minibatch.Count());
                policyNetwork.reward.Add(totalreward);
                // Apply gradients after batch processing
                
            }
            policyNetwork.ApplyGradientsAdam();
            valueNetwork.ApplyGradientsAdam();
            
            policyNetwork.save(true, true);
        }

        private Node LogGaussian(List<double> actions, List<Node> mus, List<Node> sigmas)
        {
            Node logProb = 0;
            for (int i = 0; i < actions.Count; i++)
            {
                Node diff = actions[i] - mus[i];
                logProb += -0.5 * Node.Log(2 * Math.PI * Node.Pow(sigmas[i], 2)) - Node.Pow(diff, 2) / (2 * Node.Pow(sigmas[i], 2));
            }
            return logProb;
        }

        private double SampleNormal(Random rnd)
        {
            // Box-Muller transform for Gaussian sampling
            double u1 = rnd.NextDouble();
            double u2 = rnd.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }


        Node ComputeEntropy(List<Node> sigmas)
        {
            Node entropy = 0;
            foreach (var sigma in sigmas)
            {
                entropy += 0.5 * (1 + Node.Log(2 * Math.PI)) + Node.Log(sigma);
            }
            return entropy;
        }


    }
}
