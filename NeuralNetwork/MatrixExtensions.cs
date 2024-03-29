using System;
using Molytho.Matrix;

namespace Molytho.NeuralNetwork
{
    public static class MatrixExtensions
    {
        private static Random random = new();
        public static void PopulateRandom(this Matrix<double> @this)
        {
            for (int x = 0; x < @this.Width; x++)
                for (int y = 0; y < @this.Height; y++)
                    @this[x, y] = random.NextDouble();
        }

        public static Matrix<double> RemoveBiasFromTranspose(this Matrix<double> @this)
        {
            double[,] matrix = (double[,])@this;

            return new Matrix<double>(matrix, matrix.GetLength(0) - 1);
        }
    }
}