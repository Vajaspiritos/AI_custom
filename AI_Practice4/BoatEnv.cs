using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI_Practice4
{
    public class BoatEnv : IEnviroment
    {

        float WW = 1440f;
        float WH = 740f;
        float boatwidth = 88f;
        float boatheight = 50f;
        float coinsize = 50f;
        float decay = 0.95f;
        //float angle = 0f;
        float angle = (float)Random.Shared.NextDouble() * 360 - 180;
        float turnspeed = 4f;
       static float speed = 8;
        
        float momentumY = 0f;
        float momentumX = 0f;
        float momentumTurn = 0f;
     
        bool LEFT = false;
        bool UP = false;
        bool RIGHT = false;
        float max_possible_distance = MathF.Sqrt(2 * speed * speed);
        float[] coin_locaction;
        float[] boat_locaction;
        float[] boat_vector_OG = [1, 0];
        float[] boat_vector = [1, 0];





        public double[] GetState()
        {
            double[] vectortocoin = [coin_locaction[0] - boat_locaction[0], coin_locaction[1] - boat_locaction[1]];
            float angletocoin = MathF.Atan2((float)vectortocoin[1], (float)vectortocoin[0]) * (180f / MathF.PI)%360;
            float distance = (float)Math.Sqrt((boat_locaction[0] - coin_locaction[0]) * (boat_locaction[0] - coin_locaction[0]) + (boat_locaction[1] - coin_locaction[1]) * (boat_locaction[1] - coin_locaction[1]));

            // return [boat_locaction[0], boat_locaction[1], coin_locaction[0], coin_locaction[1], angle,angletocoin, momentumTurn / 10];
            //return [angle,angletocoin, momentumTurn / 10];
            // return [boat_locaction[0] / WW, boat_locaction[1] / WH, coin_locaction[0] / WW, coin_locaction[1] / WH, boat_vector[0], boat_vector[1], momentumTurn / 10, momentumX / 10, momentumY / 10];
            //return [boat_locaction[0] , boat_locaction[1] , coin_locaction[0] , coin_locaction[1] , boat_vector[0], boat_vector[1], momentumTurn / 10, momentumX / 10, momentumY / 10];
            return [
    boat_locaction[0] / WW,
    boat_locaction[1] / WH,
    coin_locaction[0] / WW,
    coin_locaction[1] / WH,
    angle,
    angletocoin,
    momentumTurn / turnspeed,  // Normalize to max turn speed
    momentumX / speed,   // Normalize to max speed
    momentumY / speed,
    distance / max_possible_distance
];

        }

        public void Render()
        {
            throw new NotImplementedException();
        }

        public double[] Reset()
        {
             coin_locaction = [(float)Math.Floor(Random.Shared.NextDouble() * WW), (float)Math.Floor(Random.Shared.NextDouble() * WH)];
            boat_locaction = [(float)Math.Floor(Random.Shared.NextDouble() * WW), (float)Math.Floor(Random.Shared.NextDouble() * WH)];
            return GetState();
        }
        private float[] Turn(float[] vector, float angle)
        {
            float radians = angle * (MathF.PI / 180f);
            float s = MathF.Sin(radians);
            float c = MathF.Cos(radians);
            return new float[] { c * vector[0] - s * vector[1], s * vector[0] + c * vector[1] };
        }
        public (double[], double, bool) Step(double[] action)
        {


            double[] vectortocoin = [coin_locaction[0] - boat_locaction[0], coin_locaction[1] - boat_locaction[1]];


            bool done = false;
            float reward = 0f;
            float distance0 = MathF.Sqrt((boat_locaction[0] - coin_locaction[0]) * (boat_locaction[0] - coin_locaction[0]) + (boat_locaction[1] - coin_locaction[1]) * (boat_locaction[1] - coin_locaction[1]));
            if (action.Length == 1)
            {
                //Console.WriteLine(action[0]);
                if (action[0] == 0) { UP = true;    LEFT = RIGHT = false; }
                if (action[0] == 1) { LEFT = true;    UP = RIGHT = false; }
                if (action[0] == 2) { RIGHT = true; LEFT =    UP = false; }
            }
            else
            {
                UP = action[0] > 0.99f;
                LEFT = action[1] > 0.99f;
                RIGHT = action[2] > 0.99f;
            }

           // UP = false;

            if (UP) { momentumX = boat_vector[0] * speed; momentumY = boat_vector[1] * speed; }
            if (LEFT) momentumTurn = -turnspeed;
            if (RIGHT) momentumTurn = turnspeed;

            float correctedangle0 = (angle < 0) ? 360 - angle : angle;
            float[] vectortocoin0 = [coin_locaction[0] - boat_locaction[0], coin_locaction[1] - boat_locaction[1]];
            float angletocoin0 = MathF.Atan2(vectortocoin0[1], vectortocoin0[0]) * (180f / MathF.PI);
            float diff0 = angle - angletocoin0;


            if (LEFT && RIGHT) { LEFT = false; RIGHT = false; }
            momentumTurn *= decay * decay;
            momentumX *= decay;
            momentumY *= decay;
            momentumTurn = (Math.Abs(momentumTurn) < 0.01) ? 0 : momentumTurn;
            momentumX = (Math.Abs(momentumX) < 0.01) ? 0 : momentumX;
            momentumY = (Math.Abs(momentumY) < 0.01) ? 0 : momentumY;

            boat_locaction[0] += momentumX;
            boat_locaction[1] += momentumY;
            angle = (angle + momentumTurn) % 360;
            boat_vector = Turn(boat_vector_OG, angle);

            float wallpenalty = 1f;

            if (boat_locaction[0] < boatwidth / 2) { done = true; reward = -wallpenalty; boat_locaction[0] = boatwidth / 2; }
            if (boat_locaction[0] > WW - boatwidth / 2) { done = true; reward = -wallpenalty; boat_locaction[0] = WW - boatwidth / 2; }
            if (boat_locaction[1] < boatheight / 2) { done = true; reward = -wallpenalty; boat_locaction[1] = boatheight / 2; }
            if (boat_locaction[1] > WH - boatheight / 2) { done = true; reward = -wallpenalty; boat_locaction[1] = WH - boatheight / 2; }
            bool coin = false;
            float distance = (float)Math.Sqrt((boat_locaction[0] - coin_locaction[0]) * (boat_locaction[0] - coin_locaction[0]) + (boat_locaction[1] - coin_locaction[1]) * (boat_locaction[1] - coin_locaction[1]));
            if (distance < coinsize / 2)
            {
                coin = true;
                reward += 5;
                coin_locaction = [(int)Math.Floor(Random.Shared.NextDouble() * WW), (int)Math.Floor(Random.Shared.NextDouble() * WH)];
                done = true;
            }
            

            float correctedangle = (angle < 0) ? 360 - angle : angle;
            vectortocoin = [coin_locaction[0] - boat_locaction[0], coin_locaction[1] - boat_locaction[1]];
            double angletocoin = (Math.Atan2(vectortocoin[1], vectortocoin[0]) * (180f / MathF.PI))%360;
            double diff = (angle - angletocoin)%360;
            if (diff > 180f)
            {
                diff -= 360f;
            }
            else if (diff < -180f)
            {
                diff += 360f;
            }
            if ((Math.Abs(diff0) < 30) && Math.Abs(diff) > 30 && !coin){
                reward -= 1;
                done = true;
            }
                //reward += (distance0- distance)/10f;
                reward += (distance0 - distance)>0? 1f:-1f;
            reward += (Math.Abs(diff) < 30)? 1f:-1f;
            if (distance0 == distance) reward -= 1f;
            done = true;
           // if (Math.Abs(diff) > 150) { reward = -1f; done = true; }
            // if (Math.Abs(diff) > 150) done = true;
            // reward = MathF.Tanh(reward);
            //UP LEFT RIGHT

            return (GetState(), reward, done);


        }
    }
}
