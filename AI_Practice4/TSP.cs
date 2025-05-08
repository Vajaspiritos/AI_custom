using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace AI_Practice4
{
    internal class TSP : IEnviroment
    {
        public Graph g;
        public Graph g_base;
        int PC = 6;
        List<int> PointsState = new List<int>();
        int maxLength = 100;
        int Current_point = 1;
        double Distance = 0;
        const int Starter_point = 1;
        List<int> memory = new List<int>();

        
        public double[] GetState()
        {
            List<double> routes = new List<double>();
            for (int i = 1; i <= PC; i++)
            {
                for (int j = i+1; j <= PC; j++)
                {
                    routes.Add(g.getDistance(i, j)/100);
                }
            }
            foreach (int i in PointsState) { 
            routes.Add ((int)i);
            }
            return routes.ToArray();
         }

        public void Render()
        {
            Console.WriteLine($"Currently at: {Current_point}");
            

        }

        public double[] Reset()
        {
            memory.Clear();
            PointsState.Clear();
            PointsState.Add(1);
            for (int i = 0; i < PC-1; i++)
            {
                PointsState.Add(0);
            }
            Distance = 0;
            Current_point = Starter_point ;
            if (g_base != null)
            {
                g = g_base;
            }
            else
            {
                g = new Graph();
                for (int i = 1; i <= PC; i++)
                {
                    for (int j = i + 1; j <= PC; j++)
                    {
                        g.SetDistance(i, j, Random.Shared.NextDouble() * 100);

                    }

                }
            }
            return GetState();
        }
        double rewardcollector = 0;
        public (double[], double, bool) Step(double[] action)
        {
            bool done = false;
            double reward = 0;
            int DESTINATION = 0;
            DESTINATION = action.Length==0?(int)action[0]:(int)(action.Max())+1;
            DESTINATION = Math.Clamp(DESTINATION, 1, PC);
            if (memory.Contains(DESTINATION) || DESTINATION == Current_point)   { done = true;  }
            if (DESTINATION > PC || DESTINATION < 1)                            { done = true; reward = -0.01 * (Math.Abs(DESTINATION)+1); };
            if (!done) Distance =  ( maxLength- g.getDistance(Current_point, DESTINATION))/maxLength;
           
            if (DESTINATION == Starter_point) { done = true; }
            if (PC - 1 == memory.Count) { reward = Math.Abs(DESTINATION) * -0.1; }
            if (PC - 1 == memory.Count&& DESTINATION==Starter_point) { reward += 1; }
           // if (PC - 1 != memory.Count && DESTINATION == Starter_point) reward = 3;

            //  Distance += 1;
            reward += Math.Pow(Distance,1);
            reward -= 0.2*(PC - memory.Count);
               // reward -= Distance;
            
            
            
          
             reward = Math.Tanh(reward);

            memory.Add(DESTINATION);
            if (!done)
            {
               
                PointsState[Current_point - 1] = -1;
                PointsState[DESTINATION - 1] = 1;
                Current_point = DESTINATION;
            }

            string line = "";
            if (done)
            {
                line = "1";
                foreach (int Point in memory)
                {
                    line += (" -> " + Point);
                }
                String ps = "";
                foreach (int i in PointsState)
                {
                    if (i == 0) ps += "0";
                    if (i == 1) ps += "1";
                    if (i == -1) ps += "X";
                }
                Console.WriteLine(line.PadRight(40) + " |" + (rewardcollector+ reward).ToString().PadRight(20) + " |" + Distance + " | " + ps);
                rewardcollector = 0;
            }
            else rewardcollector += reward;

            return (GetState(), reward, done);
        }
   
    
    
    }
}
