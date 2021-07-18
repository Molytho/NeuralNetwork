using System;
using System.IO;
using System.Text.Json;
using Molytho.Matrix;
using Molytho.NeuralNetwork;
using Molytho.NeuralNetwork.Training;

string command = args.Length >= 1 ? args[0] : "help";

Model model;
switch (command)
{
    case "train":
        if (args.Length < 3)
        {
            Console.WriteLine("Not enough arguments!");
            goto case "help";
        }
        model = await Model.LoadFromFileAsync(args[1]);
        model.TrainFunction = Train.Default;
        string[] lines = await File.ReadAllLinesAsync(args[2]);
        int rounds = args.Length >= 4 ? int.Parse(args[3]) : 1;
        for(int i = 0; i < rounds; i++)
        {
            Console.WriteLine("Round: " + i);
            foreach(string line in lines)
            {
                string[] values = line.Split('|');
                if(values.Length != 2)
                    throw new ArgumentException("Bad trainings file");
                Vector<double> @in = (Vector<double>)JsonSerializer.Deserialize<MatrixBase<double>>(values[0]),
                    @out = (Vector<double>)JsonSerializer.Deserialize<MatrixBase<double>>(values[1]);
                model.Train(@in, @out);
            }
        }
        await model.SaveToFileAsync(args[1]);
        break;
    case "create":
        if (args.Length < 4)
        {
            Console.WriteLine("Not enough arguments!");
            goto case "help";
        }
        int[] sizes = new int[args.Length - 2];
        for (int i = 0; i < sizes.Length; i++)
            sizes[i] = int.Parse(args[i + 2]);
        model = new Model(sizes[0]);
        for (int i = 1; i < sizes.Length; i++)
            model.AddLayer(sizes[i]);
        await model.SaveToFileAsync(args[1]);
        break;
    case "info":
        if (args.Length < 2)
        {
            Console.WriteLine("Not enough arguments!");
            goto case "help";
        }
        model = await Model.LoadFromFileAsync(args[1]);
        Console.WriteLine("File: " + args[1]);
        Console.WriteLine("Input size: " + model.InputSize);
        Console.WriteLine("Output size: " + model.OutputSize);
        Console.WriteLine("Layer count: " + model.LayerCount);
        break;
    case "load":
        if (args.Length < 3)
        {
            Console.WriteLine("Not enough arguments!");
            goto case "help";
        }
        model = await Model.LoadFromFileAsync(args[1]);
        Vector<double> input = new Vector<double>(args.Length - 2);
        for (int i = 2; i < args.Length; i++)
        {
            input[i - 2] = double.Parse(args[i]);
        }
        Vector<double> result = model.Run(input);
        Console.WriteLine(result);
        break;

    default:
        Console.WriteLine("Unkown command: {0}", command);
        goto case "help";
    case "help":
        Console.WriteLine("Usage:");
        Console.WriteLine("  train FILENAME TRAINDATAFILENAME [ROUNDS]");
        Console.WriteLine("  create FILENAME INPUTSIZE [INTERMEDIATE SIZES, ...] OUTPUTSIZE");
        Console.WriteLine("  info FILENAME");
        Console.WriteLine("  load FILENAME INPUT...");
        Console.WriteLine("  help");
        return -100;
}

return 0;