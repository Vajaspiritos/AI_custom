using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI_Practice4
{
    public class Step
    {
        public List<double> State;
        public List<double> Next_State;
        public int action;
        public int Action;
        public double[] Actions;
        public double Logprobs;
        public Node[] NewProbability;
        public Node CriticEstimate;
        public double reward;
        public double Advantage;
        public double Return;
        public bool done;

    }
}
