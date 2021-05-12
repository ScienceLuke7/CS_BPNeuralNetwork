using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
	[Serializable]
	public class BackpropagationNeuralNetwork
	{
		private static readonly System.Random random = new System.Random();

		// [layer][node]
		float[][] values;
		float[][] biases;
		// [layer][weight to node][weight from node]
		float[][][] weights;

		float[][] desiredValues;
		float[][] biasesSigmoid;
		// [layer][weight to node][weight from node]
		float[][][] weightsSigmoid;

		private const float WeightDecay = 0.001f;
		private const float LearningRate = 1f;

		public BackpropagationNeuralNetwork(IReadOnlyList<int> neuralStructure)
		{
			values = new float[neuralStructure.Count][];
			biases = new float[neuralStructure.Count][];
			weights = new float[neuralStructure.Count - 1][][];
			desiredValues = new float[neuralStructure.Count][];
			biasesSigmoid = new float[neuralStructure.Count][];
			weightsSigmoid = new float[neuralStructure.Count - 1][][];

			// inits values and biases
			for (int i = 0; i < neuralStructure.Count; i++)
			{
				values[i] = new float[neuralStructure[i]];
				desiredValues[i] = new float[neuralStructure[i]];
				biases[i] = new float[neuralStructure[i]];
				biasesSigmoid[i] = new float[neuralStructure[i]];
			}

			// inits weights
			for (int i = 0; i < neuralStructure.Count - 1; i++)
			{
				weights[i] = new float[values[i + 1].Length][];
				weightsSigmoid[i] = new float[values[i + 1].Length][];
				for (int j = 0; j < weights[i].Length; j++)
				{
					weights[i][j] = new float[values[i].Length];
					weightsSigmoid[i][j] = new float[values[i].Length];
					// changes from 0 to random
					for (int k = 0; k < weights[i][j].Length; k++)
						weights[i][j][k] = (float)random.NextDouble() * MathF.Sqrt(2f / weights[i][j].Length);
				}
			}
		}

		public float[] Test(float[] inputs)
		{
			for (int i = 0; i < values[0].Length; i++)
				values[0][i] = inputs[i];
			for (int i = 1; i < values.Length; i++)
			{
				for (int j = 0; j < values[0].Length; j++)
				{
					values[i][j] = Sigmoid(Sum(values[i - 1], weights[i - 1][j]) + biases[i][j]);
					desiredValues[i][j] = values[i][j];
				}
			}
			return values[values.Length - 1];
		}

		private static float Sum(IEnumerable<float> values, IReadOnlyList<float> weights) => values.Select((v, i) => v * weights[i]).Sum();
		private static float Sigmoid(float x) => 1f / (1f + (float)Math.Exp(-x));
		private static float SigmoidDerivative(float x) => x * (1 - x); 

		public void Train(float[][] trainingInputs, float[][] trainingOutputs)
		{
			// smudge for loop
			for (int i = 0; i < trainingInputs.Length; i++)
			{
				Test(trainingInputs[i]);

				for (int j = 0; j < desiredValues[desiredValues.Length - 1].Length; j++)
					desiredValues[desiredValues.Length - 1][j] = trainingOutputs[i][j];
				for (int j = values.Length - 1; j >= 1; j--)
				{
					for (int k = 0; k < values[j].Length; k++)
					{
						var biasSmudge = SigmoidDerivative(values[j][k]) * (desiredValues[j][k] - values[j][k]);
						biasesSigmoid[j][k] += biasSmudge;
						for (int l = 0; l < values[j -1].Length; l++)
						{
							var weightSmudge = values[j - 1][l] * biasSmudge;
							weightsSigmoid[j - 1][k][l] += weightSmudge;
							var valueSmudge = weights[j - 1][k][l] * biasSmudge;
							desiredValues[j - 1][l] += valueSmudge;
						}
					}
				}

			}

			for (int i = values.Length - 1; i >= 1; i--)
			{
				for (int j = 0; j < values[i].Length; j++)
				{
					biases[i][j] += biasesSigmoid[i][j] * LearningRate;
					biases[i][j] *= 1 - WeightDecay;

					biasesSigmoid[i][j] = 0;

					for (int k = 0; k < values[i - 1].Length; k++)
					{
						weights[i - 1][j][k] += weightsSigmoid[i - 1][j][k] * LearningRate;
						weights[i - 1][j][k] *= 1 - WeightDecay;
						weightsSigmoid[i - 1][j][k] = 0;
					}

					desiredValues[i][j] = 0;
				}
			}

		}
	}
}
