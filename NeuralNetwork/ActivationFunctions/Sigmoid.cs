using System;
using Molytho.Matrix;

namespace Molytho.NeuralNetwork
{
    public static class ActivationFunctions
    {
        private static float NonVecSigmoid(float input)
            => 1 / 1 + MathF.Pow(MathF.E, -input);
        public static Vector<float> Sigmoid(Vector<float> input)
        {
            Vector<float> ret = new Vector<float>(input.Size);
            for (int i = 0; i < input.Size; i++)
            {
                ret[i] = NonVecSigmoid(input[i]);
            }
            return ret;
        }
        public static Vector<float> DifferentialSigmoid(Vector<float> input)
        {
            Vector<float> ret = new Vector<float>(input.Size);
            for (int i = 0; i < input.Size; i++)
            {
                ret[i] = NonVecSigmoid(input[i]) * (1 - NonVecSigmoid(input[i]));
            }
            return ret;
        }
    }
}
