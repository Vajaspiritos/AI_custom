using System;
using System.Collections.Generic;
using System.Linq;

namespace AI_Practice4
{
    public class PPO_Discrete
    {
        private Network policyNetwork; // Main policy network for action selection.
        private Network valueNetwork;  // Value network for estimating returns.
        private IEnviroment env;       // Environment interface.
        private double gamma = 0.99;   // Discount factor.
        private double lambda = 0.95;  // GAE lambda for smoothing advantages.
        private double clipEpsilon = 0.2; // Clipping parameter for PPO.
        private int batchSize = 256;    // Batch size for training.
        private int epochs = 5;       // Number of epochs per update.
        private int bufferCapacity = 2*4096; // Rollout buffer capacity.
        private List<Step> rolloutBuffer = new List<Step>();
        private double bestReward = 0;
        public PPO_Discrete(Network policyNetwork, Network valueNetwork, IEnviroment env)
        {
            this.policyNetwork = policyNetwork;
            this.valueNetwork = valueNetwork;
            this.env = env;
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
                Console.WriteLine("Rollouts collected.");

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

                // Get action probabilities from policy network (softmax output)
                var actionProbs = Softmax(policyNetwork.forward(step.State)).Select(X=>X.VALUE).ToArray();

                // Sample action from the probabilities
                int action = SampleDiscreteAction(actionProbs);

                // Compute log probability of the chosen action
                step.Logprobs = Math.Log(actionProbs[action]);

                // Execute action in the environment
                (double[] nextState, double reward, bool done) = env.Step(new double[] { action });
                step.Actions = new double[] { action };
                step.reward = reward;
                step.Next_State = nextState.ToList();
                step.done = done;

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
            for (int e = 0; e < epochs; e++)
            {
                Console.WriteLine($"Epoch {e + 1} / {epochs}");

                //var minibatch = rolloutBuffer.OrderBy(x => rnd.Next()).Take(batchSize).ToList();
                for (int ee = 0; ee < Math.Ceiling((double)bufferCapacity / batchSize); ee++)
                {
                    var minibatch = shuffled.Skip(ee * batchSize).Take(batchSize);
                    double totalloss = 0;
                    double totalreward = 0;
                    double entropyCoeff = 0.01;
                    foreach (var step in minibatch)
                    {
                        // Compute the new log probability
                        var newActionProbs = Softmax(policyNetwork.forward(step.State));
                        Node newLogProb = Node.Log(newActionProbs[(int)step.Actions[0]]);

                        // Compute ratio and clip
                        Node ratio = Node.Exp(newLogProb - step.Logprobs);
                        Node clippedRatio = Node.Clamp(ratio, 1 - clipEpsilon, 1 + clipEpsilon);
                        Node policyLoss = -1 * Node.Min(ratio * step.Advantage, clippedRatio * step.Advantage);

                        // Backpropagate policy loss
                        policyLoss.Backpropagate();

                        // Value update
                        Node valueLoss = Node.Pow(step.Return - valueNetwork.forward(step.State)[0], 2);
                        valueLoss.Backpropagate();

                        totalloss += valueLoss.VALUE;
                        totalreward += step.reward;
                    }

                    if (totalreward > bestReward)
                    {
                        bestReward = totalreward;
                        policyNetwork.save(false, false, "Best.js");
                    }

                    policyNetwork.loss.Add(totalloss / minibatch.Count());
                    policyNetwork.reward.Add(totalreward);
                    policyNetwork.ApplyGradientsAdam();
                    valueNetwork.ApplyGradientsAdam();
                }
                shuffled = rolloutBuffer.OrderBy(x => rnd.Next());
                
            }

            // Apply gradients after batch processing
            
            policyNetwork.save(true);
        }

        private Node[] Softmax(List<Node> logits)
        {
            Node maxLogit = logits.Max(l => l.VALUE);
            var expLogits = logits.Select(l => Node.Exp(l - maxLogit)).ToArray();
            Node sum = 0;
            foreach (Node n in expLogits) { sum += n; }
            Node sumExpLogits = sum;
            return expLogits.Select(e => e / sumExpLogits).ToArray();
        }

        private int SampleDiscreteAction(double[] probabilities)
        {
            double randomValue = new Random().NextDouble();
            double cumulative = 0;
            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (randomValue <= cumulative)
                    return i;
            }
            return probabilities.Length - 1; // Fallback in case of rounding errors
        }
    }
}
