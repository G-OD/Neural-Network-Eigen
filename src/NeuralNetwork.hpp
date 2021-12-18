#include <eigen3/Eigen/Dense>

using namespace Eigen;

class NeuralNetwork {
	int inputs, hidden, outputs;
	double learningRate;

	MatrixXd weights_ih, weights_ho;
	MatrixXd bias_h, bias_o;

public:
	NeuralNetwork(int inputs, int hidden, int outputs, double learningRate) {
		this->inputs = inputs;
		this->hidden = hidden;
		this->outputs = outputs;
		this->learningRate = learningRate;

		weights_ih = MatrixXd::Random(hidden, inputs);
		weights_ho = MatrixXd::Random(outputs, hidden);

		bias_h = MatrixXd::Zero(hidden, 1);
		bias_o = MatrixXd::Zero(outputs, 1);
	}

	void print() {
		std::cout << weights_ih << "\n\n";
		std::cout << weights_ho << "\n\n";
		std::cout << bias_h << "\n\n";
		std::cout << bias_o << "\n\n";
	}

	constexpr static double sigmoid(double x) {
		return 1 / (1 + exp(-x));
	}
	constexpr static double dsigmoid(double x) {
		return x * (1 - x);
	}

	MatrixXd feedforward(MatrixXd inputs) {
		MatrixXd hidden = (weights_ih * inputs + bias_h).unaryExpr(&sigmoid);
		MatrixXd outputs = (weights_ho * hidden + bias_o).unaryExpr(&sigmoid);
		return outputs;
	}

	MatrixXd backpropagation(MatrixXd inputs, MatrixXd targets) {
		// Feed forward
		MatrixXd hidden = (weights_ih * inputs + bias_h).unaryExpr(&sigmoid);
		MatrixXd outputs = (weights_ho * hidden + bias_o).unaryExpr(&sigmoid);

		// Backpropagation
		MatrixXd errors = targets - outputs;
		MatrixXd gradients = errors.array() * outputs.array().unaryExpr(&dsigmoid) * learningRate;
		weights_ho +=  gradients * hidden.transpose();
		bias_o += gradients;

		errors = weights_ho.transpose() * errors;
		gradients = errors.array() * hidden.array().unaryExpr(&dsigmoid) * learningRate;
		weights_ih += gradients * inputs.transpose();
		bias_h += gradients;

		return errors;
	}

	void geneticAlgorithm() {
		
	}
};