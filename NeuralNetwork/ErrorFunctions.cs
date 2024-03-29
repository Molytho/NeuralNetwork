using System;
using Molytho.Matrix;

namespace Molytho.NeuralNetwork.Training
{
    delegate Vector<double> ErrorFunctionGradient(Vector<double> actual, Vector<double> expected);

    namespace ErrorFunctions
    {
        static class MSE
        {
            public static ErrorFunctionGradient Default => Calculate;
            private static Vector<double> Calculate(Vector<double> actual, Vector<double> expected)
            {
                Vector<double> ret = new Vector<double>(actual.Dimension);

                for (int i = 0; i < actual.Height; i++)
                {
                    ret[i] = -(expected[i] - actual[i]);
                }

                return ret;
            }
        }
    }
}