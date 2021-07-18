using System;
using Molytho.Matrix;
using Molytho.NeuralNetwork;

string command = args.Length >= 1 ? args[0] : "help";

Model model;
switch (command)
{
    case "load":
        if(args.Length < 3)
        {
            Console.WriteLine("Not enough arguments!");
            goto case "help";
        }
        model = await Model.LoadFromFileAsync(args[1]);
        Vector<double> input = new Vector<double>(args.Length - 2);
        for(int i = 2; i < args.Length; i++)
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
        Console.WriteLine("  load FILENAME INPUT...");
        Console.WriteLine("  help");
        return -100;
}

return 0;