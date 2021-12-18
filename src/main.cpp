#include "stdafx.hpp"

int main() {
	srand(time(NULL));

	NeuralNetwork nn(2, 4, 1, 1);

	std::vector<Matrix<double, 2, 1>> trainingInputs;
	{
		Matrix<double, 2, 1> in;
		in(0, 0) = 1;
		in(1, 0) = 0;
		trainingInputs.push_back(in);

		in(0, 0) = 0;
		in(1, 0) = 1;
		trainingInputs.push_back(in);

		in(0, 0) = 1;
		in(1, 0) = 1;
		trainingInputs.push_back(in);

		in(0, 0) = 0;
		in(1, 0) = 0;
		trainingInputs.push_back(in);
	}
	std::vector<Matrix<double, 1, 1>> targets;
	{
		Matrix<double, 1, 1> in;

		in << 1;
		targets.push_back(in);

		in << 1;
		targets.push_back(in);

		in << 0;
		targets.push_back(in);

		in << 0;
		targets.push_back(in);
	}


	for (int i = 0; i < 50000; i++) {
		int index = i % trainingInputs.size();
		MatrixXd inputs = trainingInputs[index];
		
		MatrixXd errors = nn.backpropagation(inputs, targets[index]);
		if (i % 1000 == 0) {
			double mean = errors.cwiseAbs().mean();
			std::cout << "Error: " << mean << std::endl;
		}
	}

	MatrixXd inputs(2, 1);
	inputs << 1, 0;
	std::cout << nn.feedforward(inputs) << std::endl;
	inputs << 0, 1;
	std::cout << nn.feedforward(inputs) << std::endl;
	inputs << 1, 1;
	std::cout << nn.feedforward(inputs) << std::endl;
	inputs << 0, 0;
	std::cout << nn.feedforward(inputs) << std::endl;

	nn.print();
}
