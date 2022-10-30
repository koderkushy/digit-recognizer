#ifndef NN_RELU_H_
#define NN_RELU_H_

#include "../Operations/trivial.h"

namespace nn {

template<
	class NextLayer
>
struct ReLU {
	NextLayer L { };

	template<uint64_t N, uint64_t channels>
	auto recurse (nn::operations::image<N, channels> X, const int label) {
		auto [gradient, loss] = L.recurse(forward(X), label);
		return pair(backward(X, gradient), loss);
	}

	template<uint64_t N, uint64_t channels>
	auto evaluate (const nn::operations::image<N, channels>& X, const int label) {
		return L.evaluate(forward(X), label);
	}

	template<uint64_t N, uint64_t channels>
	auto forward (nn::operations::image<N, channels> X) {

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (X[f][i][j] < 0)
						X[f][i][j] = 0;

		return std::move(X);
	}

	auto optimize () {
		L.optimize();
	}

	template<uint64_t N, uint64_t channels>
	auto backward (const nn::operations::image<N, channels>& X, const nn::operations::image<N, channels>& grad_Y) {
		nn::operations::image<N, channels> grad_X{};

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (X[f][i][j] > 0)
						grad_X[f][i][j] = grad_Y[f][i][j];

		return std::move(grad_X);
	}

	auto save (const string path) {
		L.save(path);
	}
};

} // namespace nn

#endif // NN_RELU_H_