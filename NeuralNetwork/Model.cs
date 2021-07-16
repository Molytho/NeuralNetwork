using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Molytho.Matrix;
using Molytho.NeuralNetwork.Training;

namespace Molytho.NeuralNetwork
{
    [Serializable]
    [JsonConverter(typeof(JsonModelConverter))]
    public class Model
    {
        private readonly LinkedList<Layer> layers = new LinkedList<Layer>();
        private readonly int inSize;

        private LinkedListNode<Layer>? First => layers.First;
        private LinkedListNode<Layer>? Last => layers.Last;

        private State state;

        public TrainCallback? TrainFunction { get; }
        public bool IsTrainable => TrainFunction is not null;
        public Model(int inSize, TrainCallback? trainFunc = null)
        {
            this.TrainFunction = trainFunc;
            this.inSize = inSize;
            state = State.Creation;
        }
        private Model(int inSize, LinkedList<Layer> layers)
        {
            this.inSize = inSize;
            this.layers = layers;
            this.TrainFunction = null;
            state = State.Calculation;
        }

        public Model AddLayer(int nodeCount, bool bias = true, ActivationFunction? activationFunction = null)
        {
            if (state != State.Creation)
                throw new InvalidOperationException();

            if (activationFunction is null)
                activationFunction = ActivationFunctions.LogisticFunction.Default;

            int inSize = Last?.Value.NodeCount ?? this.inSize;
            int outSize = nodeCount;
            Layer newLayer = new Layer(activationFunction, inSize, outSize, bias);
            layers.AddLast(newLayer);

            return this;
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
            if (!IsTrainable)
                throw new InvalidOperationException();

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

                @in = @out;

                if (current.Next != null)
                    current = current.Next;
                else
                    break;
            }

            TrainFunction!(trainData, output);
        }


        public Task SaveToFileAsync(string fileName) => SaveToFileAsync(new FileStream(fileName, FileMode.Create, FileAccess.Write), true);
        public async Task SaveToFileAsync(FileStream file, bool dispose)
        {
            using StreamWriter writer = new StreamWriter(file);

            string json = JsonSerializer.Serialize(this);
            await writer.WriteAsync(json).ConfigureAwait(false);

            if (dispose)
                await file.DisposeAsync().ConfigureAwait(false);
        }
        public static Task<Model> LoadFromFileAsync(string fileName) => LoadFromFileAsync(new FileStream(fileName, FileMode.Open, FileAccess.Read), true);
        public static async Task<Model> LoadFromFileAsync(FileStream file, bool dispose = false)
        {
            using StreamReader reader = new StreamReader(file);

            string json = await reader.ReadToEndAsync();
            Model model = JsonSerializer.Deserialize<Model>(json)
                ?? throw new JsonException("json data invalid");

            if (dispose)
                await file.DisposeAsync();

            return model;
        }


        private enum State
        {
            Creation,
            Calculation
        }
        public class JsonModelConverter : JsonConverter<Model>
        {
            public override Model Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
            {
                if (reader.TokenType != JsonTokenType.StartObject)
                    throw new JsonException();

                int inSize = 0;
                LinkedList<Layer>? layers = null;
                while (reader.Read())
                {
                    if (reader.TokenType == JsonTokenType.EndObject)
                    {
                        if (inSize == 0
                            || layers is null)
                            throw new JsonException();

                        return new Model(
                            inSize,
                            layers
                        );
                    }

                    if (reader.TokenType != JsonTokenType.PropertyName)
                        throw new JsonException();
                    switch (reader.GetString())
                    {
                        case "InSize":
                            reader.Read();
                            inSize = reader.GetInt32();
                            break;
                        case "Layers":
                            reader.Read();
                            if (reader.TokenType != JsonTokenType.StartArray)
                                throw new JsonException();
                            layers = new LinkedList<Layer>();
                            while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
                            {
                                Layer layer = JsonSerializer.Deserialize<Layer>(ref reader, options)
                                    ?? throw new JsonException();
                                layers.AddLast(layer);
                            }
                            break;
                        default:
                            throw new NotSupportedException();
                    }
                }
                throw new JsonException();
            }

            public override void Write(Utf8JsonWriter writer, Model value, JsonSerializerOptions options)
            {
                writer.WriteStartObject();

                writer.WriteNumber(
                    options.PropertyNamingPolicy?.ConvertName("InSize") ?? "InSize",
                    value.inSize
                );
                writer.WriteStartArray(
                    options.PropertyNamingPolicy?.ConvertName("Layers") ?? "Layers"
                );
                LinkedListNode<Layer>? current = value.First ?? throw new NotSupportedException();
                do
                {
                    JsonSerializer.Serialize(writer, current.Value, typeof(Layer), options);
                }
                while ((current = current.Next) != null);
                writer.WriteEndArray();

                writer.WriteEndObject();
            }
        }
    }
}