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

        public static Matrix<double> RemoveBias(this Matrix<double> @this)
        {
            double[,] matrix = (double[,])@this;
            double[,] temp = new double[matrix.GetLength(0), matrix.GetLength(1) - 1];

            for (int x = 0; x < temp.GetLength(0); x++)
                for (int y = 0; y < temp.GetLength(1); y++)
                    temp[x, y] = matrix[x, y];

            return temp;
        }
    }
}