
namespace AI_Practice4
{
    public class LunarLanderEnvironment:IEnviroment
    {
        // Lander properties
        public double X { get; private set; }   // Horizontal position
        public double Y { get; private set; }   // Vertical position
        public double VX { get; private set; }  // Horizontal velocity
        public double VY { get; private set; }  // Vertical velocity
        public double Fuel { get; private set; } // Remaining fuel
        public double Angle { get; private set; } // Angle of the lander
        public double AngularVelocity { get; private set; } // Angular velocity

        // Environment properties
        private const double Gravity = -1.63; // Lunar gravity in m/s^2
        private const double MaxThrust = 4.0; // Maximum thrust force in any direction
        private const double RotationSpeed = 0.1; // Rotation speed in radians per second
        private const double LandingPadX = 0.0;
        private const double LandingPadY = 0.0;
        private const double Tolerance = 0.5; // Acceptable distance and velocity tolerance

        public LunarLanderEnvironment()
        {
            Reset();
        }

        // Resets the environment to its initial state
        public double[] Reset()
        {
            X = 0.0;
            Y = 100.0; // Start 100 meters above the surface
            VX = 0.0;
            VY = 0.0;
            Fuel = 100.0; // Start with 100 units of fuel
            Angle = 0.0;
            AngularVelocity = 0.0;
            return GetState();
        }

        // Applies an action to the environment
        public (double[], double, bool) Step(double[] action)
        {
            // Limit thrust and rotation to valid ranges
            double thrust = action[0];
            double rotation = action[1];

            thrust = Math.Clamp(thrust, 0, MaxThrust);
            rotation = Math.Clamp(rotation, -1, 1);

            if (Fuel > 0)
            {
                // Update angle and angular velocity
                AngularVelocity += rotation * RotationSpeed;
                Angle += AngularVelocity;

                // Apply thrust force
                double thrustX = thrust * Math.Sin(Angle);
                double thrustY = thrust * Math.Cos(Angle);

                // Update velocities
                VX += thrustX;
                VY += thrustY + Gravity;

                // Decrease fuel based on thrust
                Fuel -= thrust * 0.1;
            }
            else
            {
                // Apply gravity if no fuel remains
                VY += Gravity;
            }

            // Update position
            X += VX;
            Y += VY;
            return ([1],1,false);
        }

        // Returns whether the lander has landed or crashed
        public (bool Done, bool Success) CheckStatus()
        {
            if (Y <= 0)
            {
                // Check if the lander is within the landing pad tolerance and has low velocity
                bool success = Math.Abs(X - LandingPadX) <= Tolerance &&
                               Math.Abs(VX) <= Tolerance &&
                               Math.Abs(VY) <= Tolerance &&
                               Math.Abs(Angle) <= Tolerance;
                return (true, success);
            }
            return (false, false);
        }

        // Returns the current state of the environment
        public double[] GetState()
        {
            return [X, Y, VX, VY, Fuel, Angle, AngularVelocity];
        }

      

        public void Render()
        {
            throw new NotImplementedException();
        }
    }
}