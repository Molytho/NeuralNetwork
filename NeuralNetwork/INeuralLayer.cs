using System;
using Molytho.Matrix;

namespace Molytho.NeuralNetwork
{
    public interface INeuralLayer
    {
        public Vector<float> Calculate(Vector<float> input);
        public Func<Vector<float>, Vector<float>> ActivationFunction { get; }
        public int InputSize { get; }
        public int OutputSize { get; }
        public INeuralLayer? NextLayer { get; set; }
    }
}
