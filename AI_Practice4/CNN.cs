using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI_Practice4
{
    public class CNN:Layer
    {
        List<Node[,]> kernels = new List<Node[,]>();
        public CNN(int kernelCount,int kernelsize=3) {
            Random r = new Random();
            for (int i = 0; i < kernelCount; i++)
            {
                Node[,] kernel = new Node[kernelsize, kernelsize];
                for (int j = 0; j < kernelsize; j++)
                {
                    for (int k = 0; k < kernelsize; k++)
                    {
                        kernel[j,k] = (r.NextDouble()*2-1);

                    }
                }
                kernels.Add(kernel);
            }
        
        }

        public override void feed(List<double> input)
        {

            for (int i = 0; i < input.Count; i++)
            {
                Neurons[i].VALUE = input[i];
                
            }

        }

    }
}
