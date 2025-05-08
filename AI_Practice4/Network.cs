using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI_Practice4
{
    public class Network
    {
        public List<Layer> Layers = new List<Layer>();
        public double learningRate;
        public List<double> loss = new List<double>();
        public List<double> reward = new List<double>();
        public Network(List<Layer> Layers,double learningRate = 0.01d) { 
        this.Layers = Layers;
            this.learningRate = learningRate;
        }

        public List<Node> forward(List<double> input) {

            Layers[0].feed(input);

            for (int i = 1; i < Layers.Count; i++) {
                Layers[i].Forward();
            }

            return Layers[Layers.Count - 1].Neurons.ToList();
        }


        public void Test(List<double> input) { 
        
            List<Node> res = forward(input);
            Console.WriteLine(string.Join(", ", res.Select(X=>X.VALUE)));


        }

        public void save(bool Log = false,bool distribution = false,string altname=null)
        {
            if (Log)
            {
                using (StreamWriter sw2 = new StreamWriter("C:\\Users\\gugla\\Desktop\\hajo\\loss.js"))
                {
                    sw2.Write("loss = [");
                    // Uncomment and adapt the next line if you want to log losses
                    sw2.Write(string.Join(",", loss.Select(x => x.ToString().Replace(",", "."))));
                    sw2.Write("]");
                }


                using (StreamWriter sw2 = new StreamWriter("C:\\Users\\gugla\\Desktop\\hajo\\reward.js"))
                {
                    sw2.Write("rewards = [");
                    // Uncomment and adapt the next line if you want to log losses
                    sw2.Write(string.Join(",", reward.Select(x => x.ToString().Replace(",", "."))));
                    sw2.Write("]");
                }
            }

            using (StreamWriter sw = new StreamWriter("C:\\Users\\gugla\\Desktop\\hajo\\"+((altname!=null)?altname:"model.js")))
            {
                sw.WriteLine("var AI = [];");
                sw.WriteLine("var layer;");
                sw.WriteLine("var neuron;");
                sw.WriteLine("var weights;");

                for (int i = 0; i < Layers.Count; i++) // Replace foreach with for
                {
                    Layer l = Layers[i];
                    sw.WriteLine("layer = [];");

                    for (int j = 0; j < l.Neurons.Count; j++) // Replace foreach with for
                    {
                        if (l.Activation == null) l.Activation = Functions.Tanh;
                       /*
                        if (j>2&&i == Layers.Count-1) l.Activation = Functions.Softplus;
                       */
                        Node n = l.Neurons[j];
                        Node b = l.Biases[j];
                        sw.WriteLine("neuron=[];");
                        sw.WriteLine("neuron['activation']='" +((distribution&& j > ((l.Neurons.Count/2)-1) && i == Layers.Count - 1)?"Softplus" : l.Activation.Method.Name)  + "';");
                        sw.WriteLine("neuron['bias']=" + b.VALUE.ToString().Replace(',', '.') + ";");
                        sw.WriteLine("neuron['value']=0.0;");
                        sw.WriteLine("neuron['weigths']=[];");

                        for (int k = 0; k < (l.Previous==null?0:l.Previous.Neurons.Count); k++) // Replace foreach with for
                        {
                            Node Weigth = l.Weights[k][j];
                            sw.WriteLine($"neuron['weigths'].push({Weigth.VALUE.ToString().Replace(',', '.')});");
                        }

                        sw.WriteLine($"layer.push(neuron);");
                    }

                    sw.WriteLine($"AI.push(layer);");
                }
            }
        }



        public void ApplyGradientsAdam(bool ascent = false)
        {
            // Parallel loop over all layers
            Parallel.For(0, Layers.Count, layerIndex =>
            {
                Layer layer = Layers[layerIndex];

                // Parallel loop over all neurons in each layer
                Parallel.For(0, layer.Neurons.Count, neuronIndex =>
                {
                   layer.Biases[neuronIndex].ApplyGradientsAdam(learningRate);
                    for (int i = 0; i < layer.Weights.Count; i++) {
                        layer.Weights[i][neuronIndex].ApplyGradientsAdam(learningRate,ascent);
                    }
                   
                });
            });
        }

        public void ApllySGD() {
            Parallel.For(0, Layers.Count, layerIndex =>
            {
                Layer layer = Layers[layerIndex];

                // Parallel loop over all neurons in each layer
                Parallel.For(0, layer.Neurons.Count, neuronIndex =>
                {
                    layer.Biases[neuronIndex].SGD(learningRate);
                    for (int i = 0; i < layer.Weights.Count; i++)
                    {
                        layer.Weights[i][neuronIndex].SGD(learningRate);
                    }

                });
            });
        }

        public override string ToString()
        {
            string ret = "";

            for (int i = 0; i < Layers.Count; i++) {

                ret += "\n\nLayer " + i;
                ret += Layers[i].ToString();
            
            }
            return ret;
        }



    }
}
