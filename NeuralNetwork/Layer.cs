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
        public bool HasBiasNode { get; }

        public Layer(ActivationFunction activationFunction, int inputSize, int outputSize, bool hasBiasNode = true)
        {
            this.activationFunction = activationFunction;
            this.weight = new Matrix<double>(outputSize, hasBiasNode ? inputSize + 1 : inputSize);
            this.HasBiasNode = hasBiasNode;

            this.weight.PopulateRandom();
        }
        public Layer(ActivationFunction activationFunction, Matrix<Double> weight, bool hasBiasNode)
        {
            this.activationFunction = activationFunction;
            this.weight = weight;
            this.HasBiasNode = hasBiasNode;
        }

        public Vector<double> Calculate(ref Vector<double> inValue, out Vector<double> internValue)
        {
            if (HasBiasNode)
            {
                double[] valueStorage = (double[])inValue;
                Array.Resize(ref valueStorage, valueStorage.Length + 1);
                valueStorage[^1] = 1;
                inValue = (Vector<double>)valueStorage;
            }

            internValue = weight * inValue;
            return ActivationFunction(internValue);
        }
    }
}