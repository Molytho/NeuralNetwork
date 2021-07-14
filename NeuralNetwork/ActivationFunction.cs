using System;
using Molytho.Matrix;

namespace Molytho.NeuralNetwork
{
    public delegate Vector<Double> VectorFunction(Vector<Double> values);
    public record ActivationFunction(VectorFunction Function, VectorFunction Differential);

    namespace ActivationFunctions
    {
        class LogisticFunction
        {
            private static ActivationFunction? _default = null;
            public static ActivationFunction Default = _default ??= new LogisticFunction(0, 1).Get;

            private readonly double x0, k;
            public LogisticFunction(double x0, double k)
            {
                this.x0 = x0;
                this.k = k;
            }

            private ActivationFunction? cache = null;
            public ActivationFunction Get => cache ??= new ActivationFunction(
                value => {
                    Vector<double> ret = new Vector<double>(value.Dimension);
                    for(int i = 0; i < value.Height; i++)
                        ret[i] = Func(value[i], x0, k);
                    return ret;
                },
                value => {
                    Vector<double> ret = new Vector<double>(value.Dimension);
                    for(int i = 0; i < value.Height; i++)
                        ret[i] = Diff(value[i], x0, k);
                    return ret;
                }
            );

            private static double Func(double x, double x0, double k)
                => 1 / (1 + Math.Exp(-k * (x - x0)));
            private static double Diff(double x, double x0, double k)
                => k * Math.Exp(-k * (x - x0)) * Func(x, x0, k) * Func(x, x0, k);
        }
    }
}