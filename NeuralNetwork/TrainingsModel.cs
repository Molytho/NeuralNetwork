using System;
using System.Collections.Generic;
using Molytho.Matrix;

namespace Molytho.NeuralNetwork.Training
{
    public record LayerTrainData(Layer Layer, Vector<double> Input, Vector<double> Intermediate, Vector<double> Output);
    public delegate void TrainCallback(LinkedList<LayerTrainData> data, Vector<double> expected);
}