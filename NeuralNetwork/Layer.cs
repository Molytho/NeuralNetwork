using System;
using Molytho.Matrix;

namespace Molytho.NeuralNetwork
{
    public class Layer
    {
        private readonly Matrix<double> weight;
        private readonly ActivationFunction activationFunction;

        public VectorFunction ActivationFunction => activationFunction.Function;
        public VectorFunction ActivationDifferential => activationFunction.Differential;
        public int NodeCount => weight.Height;
        public Matrix<double> Weights => weight;

        public Layer(ActivationFunction activationFunction, int inputSize, int outputSize)
        {
            this.activationFunction = activationFunction;
            this.weight = new Matrix<double>(outputSize, inputSize);

            this.weight.PopulateRandom();
        }
        public Layer(ActivationFunction activationFunction, Matrix<Double> weight)
        {
            this.activationFunction = activationFunction;
            this.weight = weight;
        }

        public Vector<double> Calculate(Vector<double> inValue, out Vector<double> internValue)
        {
            internValue = weight * inValue;
            return ActivationFunction(internValue);
        }
    }
}