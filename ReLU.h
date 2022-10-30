#ifndef NN_RELU_H_
#define NN_RELU_H_

// namespace nn {

template<
	class NextLayer
>
struct ReLU {
	NextLayer L { };

	template<uint64_t N, uint64_t channels>
	auto recurse (image<N, channels> X, const int label) {
		auto [gradient, loss] = L.recurse(forward(X), label);
		return pair(backward(X, gradient), loss);
	}

	template<uint64_t N, uint64_t channels>
	auto evaluate (const image<N, channels>& X, const int label) {
		return L.evaluate(forward(X), label);
	}

	template<uint64_t N, uint64_t channels>
	auto forward (image<N, channels> X) {

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
	auto backward (const image<N, channels>& X, const image<N, channels>& grad_Y) {
		image<N, channels> grad_X{};

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

// } // namespace nn

#endif // NN_RELU_H_