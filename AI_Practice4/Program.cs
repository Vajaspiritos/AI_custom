
using AI_practice4;
using System;

namespace AI_Practice4
{
    internal class Program
    {
        static void Main(string[] args)
        {

            Graph g = new Graph();
            
               g.SetDistance(1,2,10);
               g.SetDistance(1,3,20);
               g.SetDistance(1,4,27);
               g.SetDistance(1,5,15);
               g.SetDistance(1,6,10);
               g.SetDistance(2,3,10);
               g.SetDistance(2,4,15);
               g.SetDistance(2,5,10);
               g.SetDistance(2,6,15);
               g.SetDistance(3,4,10);
               g.SetDistance(3,5,15);
               g.SetDistance(3,6,27);
               g.SetDistance(4,5,10);
               g.SetDistance(4,6,20);
               g.SetDistance(5,6,10);
               
            /*
            g.SetDistance(1, 2, 10);
            g.SetDistance(1, 3, 16);
            g.SetDistance(1, 4, 10);
            g.SetDistance(2, 3, 10);
            g.SetDistance(2, 4, 16);
            g.SetDistance(3, 4, 10);
                */            

            int n =36;
            
            Plan p = new Plan();
            p.Add(21, null);
            p.Add(n, Functions.Tanh);
            p.Add(n, Functions.Tanh);
            //p.Add(n, Functions.Tanh);
          
           // p.Add(n, Functions.Tanh);
           // p.Add(n, Functions.Tanh);
          //  p.Add(16, Functions.Tanh);
          //  p.Add(n, Functions.ReLU);
            //p.Add(16, Functions.Tanh);
            p.Add(6, Functions.Linear);

            Plan p2 = new Plan();
            p2.Add(21, null);
            p2.Add(n, Functions.Tanh);
            p2.Add(n, Functions.Tanh);
          //  p2.Add(n, Functions.Tanh);
            
           // p2.Add(n, Functions.Tanh);
            //p2.Add(n, Functions.Tanh);
           // p2.Add(16, Functions.Tanh);
           // p2.Add(n, Functions.ReLU);
            //p2.Add(16, Functions.Tanh);
            p2.Add(1, Functions.Linear);

            Network AI     = new Network(p.Build(true),0.001);
            Network CRITIC = new Network(p2.Build(true),0.001);
            TSP env = new TSP();
           // PPO agent = new PPO(AI, CRITIC, env);
            PPO_Discrete agent = new PPO_Discrete(AI,CRITIC,env);
           // env.g_base = g;
            agent.Learn(300);

            while (false)
            {
                env.Reset();
                //env.g = g;
                foreach (double v in env.GetState())
                {
                    Console.WriteLine(v);
                };
                Console.Write("1");
                int counter = 0;
                while (true)
                {
                    counter++;
                    int action = 0;
                  //  int action = (int)agent.forward(env.GetState().ToList())[0];
                    env.Step([action]);
                    
                    

                    if (action == 1 ||counter>100) break;
                }
                try
                {
                    string a = Console.ReadLine();
                    if (a.Split("l").Length !=1) agent.Learn(Convert.ToInt32(a.Split("l")[1]));
                }
                catch (Exception e) { }
            }

            /*

            Plan plan = new Plan();
            Plan plan2 = new Plan();

            plan.Add(10, null);       
            plan2.Add(10, null);
            
            //int num =32;
            //for (int i = 0; i < 2; i++) {

               // plan.Add(num,  i==0?Functions.Step: Functions.Tanh);
             //   plan2.Add(num, i==0? Functions.Step : Functions.Tanh);
            //}
            
            /*
           plan.Add(32, Functions.Step);
            plan2.Add(32, Functions.Step);
            plan.Add(16, Functions.Tanh);
            plan2.Add(16, Functions.Tanh);
           // plan.Add(8, Functions.Tanh);
           // plan2.Add(8, Functions.Tanh);
            

            plan.Add(3,Functions.Linear);
            plan2.Add(1, Functions.Linear);
            
            Network AI     = new Network(plan.Build(true) ,0.01);
            Network Critic = new Network(plan2.Build(true),0.01);
            
            //AI.Layers[AI.Layers.Count - 1].Exception.Add(3, Functions.Softplus);
             //AI.Layers[AI.Layers.Count - 1].Exception.Add(4, Functions.Softplus);
            //AI.Layers[AI.Layers.Count - 1].Exception.Add(5, Functions.Softplus);
           

           //CartPole b = new CartPole();
           BoatEnv b = new BoatEnv();
            PPO_Discrete ppo = new PPO_Discrete(AI,Critic,b);
            //DQN dqn = new DQN(AI,b);
            //Boat b = new Boat(AI,Critic,8,4);
             ppo.Learn(1000);
           // dqn.Learn(25_000);

          //  ppo.render();
            //b.Render();
           
            Console.WriteLine("Training done");
            */



        }
    }
}
