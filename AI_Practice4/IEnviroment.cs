using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI_Practice4
{
    public interface IEnviroment
    {
        public double[] GetState();
        public double[] Reset();
        public (double[], double, bool) Step(double[] action);

        public void Render();

    }
}
