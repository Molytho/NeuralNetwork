using System;
using System.Text.Json;
using System.Text.Json.Serialization;
using Molytho.Matrix;

namespace Molytho.NeuralNetwork
{
    [Serializable]
    [JsonConverter(typeof(JsonLayerConverter))]
    public class Layer
    {
        private readonly Matrix<double> weight;
        private readonly ActivationFunction activationFunction;

        public VectorFunction ActivationFunction => activationFunction.Function;
        public VectorFunction ActivationDifferential => activationFunction.Differential;
        public int NodeCount => weight.Height;
        public Matrix<double> Weights => weight;
        public bool HasBiasNode { get; }

        public Layer(ActivationFunction activationFunction, int inputSize, int outputSize, bool hasBiasNode = true)
        {
            this.activationFunction = activationFunction;
            this.weight = new Matrix<double>(outputSize, hasBiasNode ? inputSize + 1 : inputSize);
            this.HasBiasNode = hasBiasNode;

            this.weight.PopulateRandom();
        }
        public Layer(ActivationFunction activationFunction, Matrix<Double> weight, bool hasBiasNode)
        {
            this.activationFunction = activationFunction;
            this.weight = weight;
            this.HasBiasNode = hasBiasNode;
        }

        public Vector<double> Calculate(ref Vector<double> inValue, out Vector<double> internValue)
        {
            if (HasBiasNode)
            {
                double[] valueStorage = (double[])inValue;
                Array.Resize(ref valueStorage, valueStorage.Length + 1);
                valueStorage[^1] = 1;
                inValue = (Vector<double>)valueStorage;
            }

            internValue = weight * inValue;
            return ActivationFunction(internValue);
        }



        public class JsonLayerConverter : JsonConverter<Layer>
        {
            private enum LayerActivationFunction
            {
                Default
            }

            public override Layer Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
            {
                if (reader.TokenType != JsonTokenType.StartObject)
                    throw new JsonException();

                bool? hasBias = null;
                LayerActivationFunction? activationFunction = null;
                Matrix<double>? weights = null;
                while (reader.Read())
                {
                    if (reader.TokenType == JsonTokenType.EndObject)
                    {
                        if (hasBias is null
                            || activationFunction is null
                            || weights is null
                            || activationFunction > LayerActivationFunction.Default)
                            throw new JsonException();
                        
                        return new Layer(
                            activationFunction switch
                            {
                                LayerActivationFunction.Default => ActivationFunctions.LogisticFunction.Default,
                                _ => throw new NotSupportedException()
                            },
                            weights,
                            (bool)hasBias
                        );
                    }

                    if (reader.TokenType != JsonTokenType.PropertyName)
                        throw new JsonException();
                    switch (reader.GetString())
                    {
                        case "HasBiasNode":
                            reader.Read();
                            hasBias = reader.GetBoolean();
                            break;
                        case "ActivationFunction":
                            reader.Read();
                            activationFunction = (LayerActivationFunction)reader.GetInt32();
                            break;
                        case "Weights":
                            reader.Read();
                            weights = (Matrix<double>)(JsonSerializer.Deserialize(ref reader, typeof(MatrixBase<double>), options)
                                ?? throw new JsonException());
                            break;
                        default:
                            throw new NotSupportedException();
                    }
                }
                throw new JsonException();
            }
            public override void Write(Utf8JsonWriter writer, Layer value, JsonSerializerOptions options)
            {
                writer.WriteStartObject();

                writer.WriteBoolean(
                    options.PropertyNamingPolicy?.ConvertName("HasBiasNode") ?? "HasBiasNode",
                    value.HasBiasNode
                );
                writer.WriteNumber(
                    options.PropertyNamingPolicy?.ConvertName("ActivationFunction") ?? "ActivationFunction",
                    value.activationFunction == ActivationFunctions.LogisticFunction.Default
                        ? (int)LayerActivationFunction.Default
                        : throw new NotSupportedException()
                );
                writer.WritePropertyName(
                    options.PropertyNamingPolicy?.ConvertName("Weights") ?? "Weights"
                );
                JsonSerializer.Serialize(writer, value.Weights, typeof(MatrixBase<double>), options);

                writer.WriteEndObject();
            }
        }
    }
}