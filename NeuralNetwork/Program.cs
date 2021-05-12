using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
	class Program
	{
		static void Main(string[] args)
		{
			Console.WriteLine("Hello World!\n");
			List<int> neuralStructure = new List<int>();
			neuralStructure.Add(2);
			neuralStructure.Add(3);
			neuralStructure.Add(2);
			BackpropagationNeuralNetwork bpnn = new BackpropagationNeuralNetwork(neuralStructure);
			float[][] inputArray = new float[][] { 
				new float[] { 1, 50 }, 
				new float[] { 10, 21, 25 },
				new float[] { 20, 60 }
			};
			float[][] outputArray = new float[][] { 
				new float[] { 2, 100 }, 
				new float[] { 20, 42, 50 },
				new float[] { 40, 120 }
			};
			for (int iteration = 0; iteration < 100; iteration++)
			{
				bpnn.Train(inputArray, outputArray);
			}
		}
	}
}
