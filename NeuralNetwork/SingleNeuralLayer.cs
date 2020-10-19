using System;
using Molytho.Matrix;

namespace Molytho.NeuralNetwork
{
    public class SingleNeuralLayer : ITrainableNeuralLayer
    {
        private readonly Matrix<float> _weights;
        private Vector<float>? lastInput, lastOutput;
        private INeuralLayer? nextLayer;

        public SingleNeuralLayer(int inputSize, int outputSize)
        {
            _weights = new Matrix<float>(outputSize, inputSize);
        }

        public Matrix<float> Weights => _weights;
        public Vector<float> LastInput => lastInput ?? throw new InvalidOperationException("Cannot get last input without using Calculate at lease once");
        public Vector<float> LastOutput => lastOutput ?? throw new InvalidOperationException("Cannot get last output without using Calculate at lease once");
        public Func<Vector<float>, Vector<float>> DifferentialActivationFunction => ActivationFunctions.DifferentialSigmoid;

        public Func<Vector<float>, Vector<float>> ActivationFunction => ActivationFunctions.Sigmoid;
        public int InputSize => _weights.Width;
        public int OutputSize => _weights.Height;
        public INeuralLayer? NextLayer { get => nextLayer; set => nextLayer = value; }

        private Vector<float> Propagate(Vector<float> input)
            => (lastOutput = ActivationFunction(_weights * (lastInput = input)));
        public Vector<float> Calculate(Vector<float> input)
            => NextLayer is null
               ? Propagate(input)
               : NextLayer.Calculate(Propagate(input));
    }
}
