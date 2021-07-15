using System;
using System.Collections.Generic;
using Molytho.Matrix;

namespace Molytho.NeuralNetwork.Training
{
    public record LayerTrainData(Layer Layer, Vector<double> Input, Vector<double> Intermediate, Vector<double> Output);
    public delegate void TrainCallback(LinkedList<LayerTrainData> data, Vector<double> expected);

    public static class Train
    {
        public static TrainCallback Default = (data, expected) => Impl(data, expected, ErrorFunctions.MSE.Default, 0.5);
        internal static void Impl(LinkedList<LayerTrainData> data, Vector<double> expected, ErrorFunctionGradient errorFunction, double learningRate)
        {
            LinkedListNode<LayerTrainData> current = data.Last ?? throw new ArgumentException(nameof(data));
            Vector<double> auxiliaryQuantity =
                errorFunction(current.Value.Output, expected)
                    .MultiplyForEach(current.Value.Layer.ActivationDifferential(current.Value.Intermediate));

            while (true)
            {
                MatrixBase<double> gradWeights = auxiliaryQuantity * current.Value.Input.Transpose;

                if (current.Previous != null)
                {
                    Matrix<double> transformationMatrix;
                    if (current.Value.Layer.HasBiasNode)
                        transformationMatrix = current.Value.Layer.Weights.RemoveBias();
                    else
                        transformationMatrix = current.Value.Layer.Weights;
                    auxiliaryQuantity = (transformationMatrix.Transpose * auxiliaryQuantity).MultiplyForEach(current.Previous.Value.Layer.ActivationDifferential(current.Previous.Value.Intermediate));
                }

                current.Value.Layer.Weights.Add(-learningRate * gradWeights);

                if (current.Previous == null)
                    break;

                current = current.Previous;
            }
        }
    }
}