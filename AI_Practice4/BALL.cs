using System;
using System.Collections.Generic;
using System.Linq;

namespace AI_Practice4
{
    internal class BALL
    {
        Network AI;

        public BALL(Network ai)
        {
            this.AI = ai;
        }

        // The Play method collects episodes and returns them for training
        public List<Step> Play(double epsilon, bool InTraining = true)
        {
            int stopAfter = (InTraining) ? 10 : -1;

            List<Step> Episode = new List<Step>();

            Random r = new Random();
            int X = r.Next(0, 101);
            int GOAL = r.Next(0, 101);

            while (stopAfter != 0)
            {
                stopAfter--;

                double reward = 0;
                Step step = new Step();
                step.State = new List<double> { X / 100.0, GOAL / 100.0 };
                bool LEFT = false;
                bool RIGHT = false;

                // Select action using epsilon-greedy policy
                List<Node> actions = Functions.Softmax(AI.forward(step.State).ToArray());
                int action;
                if (Random.Shared.NextDouble() < epsilon) // Explore
                {
                    action = Random.Shared.Next(actions.Count);
                }
                else // Exploit
                {
                    action = actions.IndexOf(actions.Max());
                }

                LEFT = action == 0;
                RIGHT = action == 1;

                // Take action and update X
                int dis0 = Math.Abs(X - GOAL);
                if (LEFT) X--;
                if (RIGHT) X++;

                // Handle boundary conditions
                if (X < 0)
                {
                    X = 0;
                    reward -= 0.2d;
                }
                if (X > 100)
                {
                    X = 100;
                    reward -= 0.2d;
                }
                if (X == GOAL)
                {
                    GOAL = r.Next(0, 101);
                    reward += 0.5;
                }
                int dis1 = Math.Abs(X - GOAL);
                reward += (dis1<dis0)?0.1:-0.1;
                reward = Math.Tanh(reward);


                step.Next_State = new List<double> { X / 100.0, GOAL / 100.0 };
                step.reward = reward;
                step.action = action;

                // Add step to episode
                Episode.Add(step);
            }

            return Episode;
        }

        // Train the network with the experience replay buffer
        public void Train(int episodes, int batchSize, double epsilonStart = 1.0, double epsilonEnd = 0.1, double epsilonDecay = 0.995)
        {
            Random random = new Random();
            double epsilon = epsilonStart; // Initial epsilon (for exploration-exploitation)
            List<Step> experienceReplayBuffer = new List<Step>(); // Buffer to store experiences

            for (int i = 0; i < episodes; i++)
            {
                // Collect episode experiences
                experienceReplayBuffer.AddRange(Play(epsilon));

                // After every episode, train the model if the replay buffer has enough samples
                if (experienceReplayBuffer.Count >= batchSize)
                {
                    // Sample a random batch from the experience buffer
                    List<Step> batch = experienceReplayBuffer.OrderBy(x => random.Next()).Take(batchSize).ToList();

                    // Train the network using the sampled batch
                    foreach (var step in batch)
                    {
                        // Calculate the target value using the Q-learning update rule
                        List<Node> nextQValues = Functions.Softmax(AI.forward(step.Next_State).ToArray());
                        double target = step.reward + 0.99 * nextQValues.Max(n => n.VALUE);

                        // Get current Q-value for the selected action
                        List<Node> currentQValues = Functions.Softmax(AI.forward(step.State).ToArray());
                        Node currentQValue = currentQValues[step.action];

                        // Calculate loss (Mean Squared Error)
                        Node targetNode = new Node(target); // Convert target to a Node
                        Node loss = Functions.MeanSquaredError(new List<Node> { currentQValue }, new List<Node> { targetNode });

                        // Backpropagate the loss and update the network
                        loss.Backpropagate();
                        AI.ApplyGradientsAdam();
                    }

                    // Reduce epsilon to encourage exploitation over time
                    epsilon = Math.Max(epsilonEnd, epsilon * epsilonDecay);
                }
            }
        }
    }
}
