using AI_Practice4;
using System;
using System.Drawing;
namespace AI_practice4
{
    public class CartPole:IEnviroment
    {
        // Constants
        private const double Gravity = 9.8;
        private const double MassCart = 1.0;
        private const double MassPole = 0.1;
        private const double TotalMass = MassCart + MassPole;
        private const double Length = 0.5; // Half of the pole length
        private const double PoleMassLength = MassPole * Length;
        private const double ForceMag = 10.0;
        private const double Tau = 0.02; // Time step

        private const double ThetaThresholdRadians = 12 * Math.PI / 180; // 12 degrees
        private const double XThreshold = 2.4; // Cart position range

        // State variables
        public double[] State { get; private set; }
        private Random _random;

        public double[] GetState() { 
        return State;
        }
        public CartPole()
        {
            _random = new Random();
            Reset();
        }

        // Resets the environment
        public double[] Reset()
        {
            State = new double[]
            {
            _random.NextDouble() * 0.1 - 0.05, // x
            _random.NextDouble() * 0.1 - 0.05, // x_dot
            _random.NextDouble() * 0.1 - 0.05, // theta
            _random.NextDouble() * 0.1 - 0.05  // theta_dot
            };
            return State;
        }

        // Steps through the environment
        public (double[], double, bool) Step(double[] action)
        {
            double x = State[0];
            double xDot = State[1];
            double theta = State[2];
            double thetaDot = State[3];

            // Apply force based on action

            int act =0;
            if (action.Length == 1)
            {
                act = (int)action[0];
            }
            else act = (action[0] > action[1]) ? 0 : 1;



            double force = act == 1 ? ForceMag : -ForceMag;

            // Equations of motion
            double costheta = Math.Cos(theta);
            double sintheta = Math.Sin(theta);

            double temp = (force + PoleMassLength * thetaDot * thetaDot * sintheta) / TotalMass;
            double thetaAcc = (Gravity * sintheta - costheta * temp) /
                              (Length * (4.0 / 3.0 - MassPole * costheta * costheta / TotalMass));
            double xAcc = temp - PoleMassLength * thetaAcc * costheta / TotalMass;

            // Update state
            x += Tau * xDot;
            xDot += Tau * xAcc;
            theta += Tau * thetaDot;
            thetaDot += Tau * thetaAcc;

            State = new double[] { x, xDot, theta, thetaDot };

            // Check if the episode is done
            bool done = x < -XThreshold || x > XThreshold ||
                        theta < -ThetaThresholdRadians || theta > ThetaThresholdRadians;

            // Reward
            double reward = done ? 0.0 : 1.0;

            return (State, reward, done);
        }

        // Render function (if needed, can be expanded)

        public void Render()
        {
            int cartWidth = 10; // Width of the cart in ASCII
            int cartHeight = 1; // Height of the cart
            int poleLength = 10; // Length of the pole in ASCII

            // Cart's x position and pole's angle
            int cartPosition = (int)(State[0] * 10); // Scale cart's x position
            int poleAngle = (int)(State[2] * 10); // Scale pole angle for simplicity

            Console.Clear(); // Clear the console for the next frame

            // Render the environment (cart + pole)
            for (int i = 0; i < 20; i++)
            {
                if (i == 10) // Cart position is on line 10
                {
                    // Draw the cart (represented by a series of characters)
                    for (int j = 0; j < cartPosition; j++)
                    {
                        Console.Write(" "); // Space before the cart
                    }
                    Console.Write("[=]"); // The cart itself
                    for (int j = 0; j < 20 - cartPosition; j++)
                    {
                        Console.Write(" "); // Space after the cart
                    }
                }

                if (i == 5) // Render the pole's angle (simplified)
                {
                    Console.Write(new string(' ', cartPosition + 5));
                    Console.WriteLine("|"); // Pole
                }
                else
                {
                    Console.WriteLine(); // Empty lines
                }
            }

            // Render the axis and other environment boundaries
            Console.WriteLine(new string('-', 40)); // Bottom boundary (ground)

            // Add any other visual elements like boundaries or additional state info
            Console.WriteLine($"Cart Position: {State[0]:F2}, Pole Angle: {State[2]:F2}");
            Console.WriteLine($"Pole Length: {poleLength}, Cart Width: {cartWidth}");
        }



    }

}
