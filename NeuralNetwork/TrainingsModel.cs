using System;
using System.Collections.Generic;
using Molytho.Matrix;

namespace Molytho.NeuralNetwork.Training
{
    public record LayerTrainData(Layer Layer, Vector<double> Input, Vector<double> Intermediate, Vector<double> Output);
    public delegate void TrainCallback(LinkedList<LayerTrainData> data, Vector<double> expected);

    public static class Train
    {
        public static TrainCallback WithLearningRate(double learningRate)
            => (data, expected) => Impl(data, expected, ErrorFunctions.MSE.Default, learningRate);
        public static TrainCallback Default = (data, expected) => Impl(data, expected, ErrorFunctions.MSE.Default, 0.5);
        internal static void Impl(LinkedList<LayerTrainData> data, Vector<double> expected, ErrorFunctionGradient errorFunction, double learningRate)
        {
            LinkedListNode<LayerTrainData>? current = data.Last ?? throw new ArgumentException(nameof(data));
            Vector<double> auxiliaryQuantity =                                              //Recursivly defined as: l = index of layer (0 <= l <= L)
                                                                                            //auxiliaryQuantity_l-1 = f'_l-1 * W_l * auxiliaryQuantity_l
                                                                                            //First instance auxiliaryQuantity_L = f' * ∇C
                  current.Value.Layer.ActivationDifferential(current.Value.Intermediate)    //This is f'
                * errorFunction(current.Value.Output, expected);                            //This is ∇C

            do
            {
                LayerTrainData currentLayer = current.Value;
                MatrixBase<double> gradWeights = auxiliaryQuantity * currentLayer.Input.Transpose;  //The chain rule of backpropagation is
                                                                                                    //Δw_ij = -n * auxiliaryQuantity_j * ∂net_j/∂w_ij
                                                                                                    //∂net_j/∂w_ij = o_j
                                                                                                    //which is just input of the layer
                if (current.Previous != null)
                {
                    LayerTrainData previosLayer = previosLayer = current.Previous.Value;
                    Matrix<double> transformationMatrix =
                        currentLayer.Layer.HasBiasNode
                        ? transformationMatrix = ((Matrix<double>)currentLayer.Layer.Weights.Transpose).RemoveBiasFromTranspose()
                        : transformationMatrix = ((Matrix<double>)currentLayer.Layer.Weights.Transpose);

                    //auxiliaryQuantity_l-1 = f'_l-1 * W_l * auxiliaryQuantity_l
                    auxiliaryQuantity = previosLayer.Layer.ActivationDifferential(previosLayer.Intermediate) * (transformationMatrix * auxiliaryQuantity);
                }

                currentLayer.Layer.Weights.Add(-learningRate * gradWeights); //We calculated the next layer so we can change weights now
            } while ((current = current.Previous) != null);
        }
    }
}