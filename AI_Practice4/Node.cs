using System;
using System.Collections.Generic;

namespace AI_Practice4
{
    public class Node : IComparable<Node>
    {
        public double VALUE;
        public double GRADIENT = 0;
        public List<Node> INPUTS;
        public List<double> GRADIENTS;

        private double m = 0; // First moment
        private double v = 0; // Second moment
        private int t = 0;    // Time step
        
        public Node(double x)
        {
            this.VALUE = x;
            INPUTS = new List<Node>();
            GRADIENTS = new List<double>();
        }
        public static implicit operator Node(double x) { 
        return new Node(x);
        }

        public Node(float x)
        {
            this.VALUE = x;
            INPUTS = new List<Node>();
            GRADIENTS = new List<double>();
        }

        public Node(int x)
        {
            this.VALUE = x;
            INPUTS = new List<Node>();
            GRADIENTS = new List<double>();
        }

        public Node(double V, List<Node> Parents, List<double> Gradients)
        {
            this.VALUE = V;
            this.INPUTS = Parents;
            this.GRADIENTS = Gradients;
        }

        public static Node operator +(Node a, Node b)
        {
            double value = a.VALUE + b.VALUE;
            return new Node(value, new List<Node> { a, b }, new List<double> { 1.0, 1.0 });
        }

        public static Node operator -(Node a, Node b)
        {
            double value = a.VALUE - b.VALUE;
            return new Node(value, new List<Node> { a, b }, new List<double> { 1.0, -1.0 });
        }

        
        public static Node operator *(Node a, Node b)
        {
            double value = a.VALUE * b.VALUE;
            return new Node(value, new List<Node> { a, b }, new List<double> { b.VALUE, a.VALUE });
        }

        public static Node[] operator *(Node a, Node[] b)
        {
            Node[] c = new Node[b.Length];
            for (int i = 0; i < b.Length; i++){
                c[i] = a * b[i];
            }


            return c;
        }

        public static bool operator >(Node a, Node b)
        {
            return a.VALUE > b.VALUE;
        }
        public static bool operator <(Node a, Node b)
        {
            return a.VALUE < b.VALUE;
        }
        public int CompareTo(Node other)
        {
            if (other == null)
                return 1; // This object is considered greater than null

            // Compare the VALUE of the current object to the VALUE of the other object
            return this.VALUE.CompareTo(other.VALUE);
        }

        // Overload Division
        public static Node operator /(Node a, Node b)
        {
            if (b.VALUE == 0)
                b.v = 1e-8;

            double value = a.VALUE / b.VALUE;
            return new Node(value, new List<Node> { a, b }, new List<double> { 1.0 / b.VALUE, -a.VALUE / (b.VALUE * b.VALUE) });
        }
        public static Node Exp(Node a)
        {
            double value = Math.Exp(a.VALUE);
            return new Node(value, new List<Node> { a }, new List<double> {value });
        }

        public static Node Log(Node a)
        {
            if (a.VALUE <= 0)
                throw new ArgumentException("Logarithm is undefined for non-positive values.");

            double value = Math.Log(a.VALUE);
            return new Node(value, new List<Node> { a }, new List<double> { 1.0 / a.VALUE });
        }

        public static Node Pow(Node a, double exponent)
        {
            double value = Math.Pow(a.VALUE, exponent);
            return new Node(value, new List<Node> { a }, new List<double> { exponent * Math.Pow(a.VALUE, exponent - 1) });
        }

        public static Node Clamp(Node a, double minValue, double maxValue)
        {
            double value = Math.Max(minValue, Math.Min(maxValue, a.VALUE));
            double gradA = (a.VALUE > minValue && a.VALUE < maxValue) ? 1.0 : 0.0;
            return new Node(value, new List<Node> { a }, new List<double> { gradA });
        }
        public static Node Max(Node a, Node b)
        {
            double value = Math.Max(a.VALUE, b.VALUE);
            double gradA = a.VALUE > b.VALUE ? 1.0 : 0.0;
            double gradB = b.VALUE >= a.VALUE ? 1.0 : 0.0;
            return new Node(value, new List<Node> { a, b }, new List<double> { gradA, gradB });
        }
        public static Node Min(Node a, Node b)
        {
            double value = Math.Min(a.VALUE, b.VALUE);
            double gradA = a.VALUE < b.VALUE ? 1.0 : 0.0;
            double gradB = b.VALUE <= a.VALUE ? 1.0 : 0.0;
            return new Node(value, new List<Node> { a, b }, new List<double> { gradA, gradB });
        }


        public void Backpropagate(double upstreamGradient = 1.0)
        {
            this.GRADIENT += upstreamGradient;

            for (int i = 0; i < INPUTS.Count; i++)
            {
                INPUTS[i].Backpropagate(upstreamGradient * GRADIENTS[i]);
            }
        }

        public void ApplyGradientsAdam(double learningRate,bool ascent=false, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        {
            t += 1;

            // Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * GRADIENT;

            // Update biased second moment estimate
            v = beta2 * v + (1 - beta2) * (GRADIENT * GRADIENT);

            // Compute bias-corrected first and second moment estimates
            double mHat = m / (1 - Math.Pow(beta1, t));
            double vHat = v / (1 - Math.Pow(beta2, t));

            // Update the value
            double delta = learningRate * mHat / (Math.Sqrt(vHat) + epsilon);
            VALUE -= ascent? -delta:delta;

            // Reset the gradient
            //Console.WriteLine(GRADIENT);
            GRADIENT = 0;
        }

        public void SGD(double learningRate) {
            VALUE -= learningRate * GRADIENT;
            GRADIENT = 0;
        }

        public override string ToString()
        {
            return $"Node(VALUE: {VALUE}, GRADIENT: {GRADIENT})";
        }
    }
}
