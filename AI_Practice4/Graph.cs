using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI_Practice4
{
    public class Graph
    {

        public Dictionary<(int,int),double> Routes = new Dictionary<(int,int),double>();
        public Dictionary<int,List<int>> Points = new Dictionary<int, List<int>>();

        public void SetDistance(int i1, int i2, double L) { 
            int min = Math.Min(i1, i2);
            int max = Math.Max(i1, i2);
            Routes[(min,max)] = L;
            if(!Points.ContainsKey(i1)) Points.Add(i1, new List<int>() { i2});
            if(!Points.ContainsKey(i2)) Points.Add(i2, new List<int>() { i1});
        }

        public double getDistance(int i1, int i2) {
            int min = Math.Min(i1, i2);
            int max = Math.Max(i1, i2);
            return Routes[(min,max)];
        }

        public List<int> getRoutes(int point) { 
        return Points[(int)point];
        }
    }
}
