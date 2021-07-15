using System;
using System.Collections.Generic;

using Molytho.Matrix;
using Molytho.NeuralNetwork.Training;

namespace Molytho.NeuralNetwork
{
    public class Model
    {
        private readonly LinkedList<Layer> layers = new LinkedList<Layer>();
        private readonly int inSize;
        private readonly TrainCallback trainFunction;

        private LinkedListNode<Layer>? First => layers.First;
        private LinkedListNode<Layer>? Last => layers.Last;

        private State state;

        public bool IsTrainable { get; }
        public Model(int inSize, TrainCallback trainFunc)
        {
            this.trainFunction = trainFunc;
            this.inSize = inSize;
            state = State.Creation;
        }

        public void AddLayer(int nodeCount)
        {
            if (state != State.Creation)
                throw new InvalidOperationException();

            int inSize = Last?.Value.NodeCount ?? this.inSize;
            int outSize = nodeCount;
            Layer newLayer = new Layer(ActivationFunctions.LogisticFunction.Default, inSize, outSize);
            layers.AddLast(newLayer);
        }

        private void CheckState()
        {
            if (state == State.Creation)
            {
                if (First != null)
                    state = State.Calculation;
                else
                    throw new InvalidOperationException();
            }
        }
        public Vector<double> Run(Vector<double> inValues)
        {
            CheckState();

            Vector<double> temp = inValues;
            LinkedListNode<Layer> current = First!;
            while (true)
            {
                Layer layer = current.Value;
                temp = layer.Calculate(ref temp, out var noOp); //noOp is only need for training
                                                                //ref temp might be changed when the layer contains a bias but this doesn't matter

                if (current.Next != null)
                    current = current.Next;
                else
                    break;
            }

            return temp;
        }
        public void Train(Vector<double> input, Vector<double> output)
        {
            CheckState();

            LinkedList<LayerTrainData> trainData = new LinkedList<LayerTrainData>();
            Vector<double> @in = input, inter, @out;
            LinkedListNode<Layer> current = First!;

            while (true)
            {
                Layer layer = current.Value;
                @out = layer.Calculate(ref @in, out inter);

                LayerTrainData layerData = new LayerTrainData(layer, @in, inter, @out);
                trainData.AddLast(layerData);

                @out = @in;

                if (current.Next != null)
                    current = current.Next;
                else
                    break;
            }

            trainFunction(trainData, output);
        }

        private enum State
        {
            Creation,
            Calculation
        }
    }
}