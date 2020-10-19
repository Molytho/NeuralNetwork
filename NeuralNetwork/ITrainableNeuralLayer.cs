using System;
using Molytho.Matrix;

namespace Molytho.NeuralNetwork
{
    public interface ITrainableNeuralLayer : INeuralLayer
    {
        public Matrix<float> Weights { get; }
        public Vector<float> LastInput { get; }
        public Vector<float> LastOutput { get; }
        public Func<Vector<float>, Vector<float>> DifferentialActivationFunction { get; }
    }
}
