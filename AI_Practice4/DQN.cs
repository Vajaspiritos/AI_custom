using System;
using System.Collections.Generic;
using System.Linq;

namespace AI_Practice4
{
    public class DQN
    {
        Network AI;
        IEnviroment env;
        double gamma = 0.99;
        int batchSize = 32;
        List<Step> replayBuffer = new List<Step>();
        int replayBufferSize = 10000; // Limit buffer size for efficiency.
        double epsilon = 0.4;
        double minepsilon = 0.1;
        double epsilondecay = 0.99;
        int maxstep = 2028;
        public DQN(Network ai, IEnviroment env)
        {
            AI = ai;
            this.env = env;
        }

        public void Learn(int episodes)
        {
            for (int i = 0; i < episodes; i++)
            {
                env.Reset();
                Console.Write($"\r{Math.Floor((double)i / (double)episodes * 10000)/100}% Done.");
                // List<Step> episode = new List<Step>();

                // Collect transitions in the episode
                int counter = 0;
                while (counter < maxstep)
                {
                    counter++;
                    Step step = new Step();
                    step.State = env.GetState().ToList();
                    var qValues = AI.forward(step.State); // No softmax, use raw Q-values.

                    // Choose action using epsilon-greedy
                    int action = ChooseAction(qValues, epsilon);

                    // Execute action and observe results
                    (double[] nextState, double reward, bool done) = env.Step([action]);
                    step.Action = action;
                    step.Next_State = nextState.ToList();
                    step.reward = reward;
                    step.done = done;

                    // Add to replay buffer
                    replayBuffer.Add(step);
                    if (replayBuffer.Count > replayBufferSize)
                        replayBuffer.RemoveAt(0); // Maintain buffer size.

                    if (done) break; // End episode if done.
                }

                // Train the network with a minibatch
                if (replayBuffer.Count >= batchSize)
                {
                    TrainNetwork();
                }
            }
        }

        private int ChooseAction(List<Node> qValues, double epsilon)
        {
            Random rnd = new Random();
            if (rnd.NextDouble() < epsilon)
            {
                // Exploration: Choose random action
                return rnd.Next(qValues.Count);
            }
            else
            {
                // Exploitation: Choose action with highest Q-value
                return qValues.IndexOf(qValues.Max());
            }
        }

        private void TrainNetwork()
        {
            // Sample a minibatch from the replay buffer
            Random rnd = new Random();
            var batch = replayBuffer.OrderBy(x => rnd.Next()).Take(batchSize).ToList();
            double totalloss = 0;
            double totalreward = 0;
            foreach (var step in batch)
            {
                Node target;
                if (step.done)
                {
                    target = step.reward; // No future reward if done.
                }
                else
                {
                    var nextQValues = AI.forward(step.Next_State);
                    target = step.reward + gamma * nextQValues.Max(); // Bellman equation.
                }

                // Compute the loss and update the network
                
                Node error = target - AI.forward(step.State)[step.Action];
                Node loss = error * error;
                totalloss += loss.VALUE;
                totalreward += step.reward;
                loss.Backpropagate();
               // AI.ApplyGradientsAdam();// Adjust weights based on error.
            }

            AI.ApplyGradientsAdam();
            AI.loss.Add(totalloss/batch.Count);
            AI.reward.Add(totalreward);

            AI.save(true);
            epsilon = Math.Max(epsilon * epsilondecay, minepsilon);
        }
        
    }

    
}
