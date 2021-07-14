using System;
using System.Collections.Generic;
using Molytho.Matrix;

namespace Molytho.NeuralNetwork
{
    public class Model
    {
        private readonly LinkedList<Layer> layers = new LinkedList<Layer>();
        private readonly int inSize;

        private LinkedListNode<Layer>? First => layers.First;
        private LinkedListNode<Layer>? Last => layers.Last;

        private State state;

        public bool IsTrainable { get; }
        public Model(int inSize)
        {
            #error This is nonsense without a trainings function
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

        public Vector<double> Run(Vector<double> inValues)
        {
            if (state == State.Creation)
            {
                if (First != null)
                    state = State.Calculation;
                else
                    throw new InvalidOperationException();
            }

            Vector<double> temp = inValues;
            LinkedListNode<Layer> current = First!;
            while (true)
            {
                Layer layer = current.Value;
                temp = layer.Calculate(temp, out var noOp); //noOp is only need for training

                if (current.Next != null)
                    current = current.Next;
                else
                    break;
            }

            return temp;
        }

        private enum State
        {
            Creation,
            Calculation
        }
    }
}