template<
	class Optimizer,
	int N,
	int in_channels,
	int M,
	int out_channels
>
struct FullyConnectedLayer {
	static constexpr int IN = N * N * in_channels, OUT = M * M * out_channels;

	array<array<double, IN>, OUT> W { };
	array<double, OUT> b { };
	vector<double> cache { };

	FullyConnectedLayer () {
		// Kaiming He initialisation

		mt19937 rng(chrono::high_resolution_clock::now().time_since_epoch().count());
		std::normal_distribution<double>
			weight_gaussian { 0, sqrt(2.0 / IN) },
			bias_gaussian {0, sqrt(2.0 / 10)};

		for (int i = 0; i < OUT; i++)
			for (int j = 0; j < IN; j++)
				W[i][j] = weight_gaussian(rng);

		for (int i = 0; i < OUT; i++)
			b[i] = bias_gaussian(rng);
	}

	auto forward (const image<N, in_channels>& X) {
		auto arr_X { array_converted(X) };
		array<double, OUT> arr_Y { };

		for (int i = 0; i < OUT; i++)
			for (int j = 0; j < IN; j++)
				arr_Y[i] += arr_X[j] * W[i][j] + b[i];

		return std::move(imagify<M, out_channels, OUT>(arr_Y));
	}

	auto train (const image<N, in_channels>& X) {
		copy_to_vector(X, cache);
		return forward(X);
	}

	auto backward (const image<M, out_channels>& grad_Y) {
		static array<array<Optimizer, IN>, OUT> W_optimizer { };
		static array<Optimizer, OUT> b_optimizer { };

		auto arr_grad_Y { array_converted(grad_Y) };
		auto arr_X { array_converted(imagify<N, in_channels>(cache)) };

		array<double, IN> arr_grad_X { };
		decltype(W) grad_W { };

		// Computing gradients wrt b
		const auto& grad_b { arr_grad_Y };

		// Computing gradients wrt X, W and optimizing
		for (int i = 0; i < OUT; i++) {
			b_optimizer[i].optimize(b[i], grad_b[i]);

			for (int j = 0; j < IN; j++)
				arr_grad_X[j] += arr_grad_Y[i] * W[i][j],
				grad_W[i][j] = arr_grad_Y[i] * arr_X[j],
				W_optimizer[i][j].optimize(W[i][j], grad_W[i][j]);
		}

		return imagify<N, in_channels, IN>(arr_grad_X);
	}

	void save (const string path) {
		ofstream out(path);
		out << fixed << setprecision(10);

		for (int i = 0; i < OUT; i++)
			for (int j = 0; j < IN; j++)
				out << W[i][j] << ",";

	}
};