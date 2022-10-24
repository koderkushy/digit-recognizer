template<
	class Optimizer,
	int N,
	int in_channels,
	int M,
	int out_channels
>
struct FullyConnectedLayer {


	array<array<double, N * N * in_channels>, M * M * out_channels> W{};
	// array<array<array<image<N, in_channels>, M>, M>, out_channels> W{};
	vector<double> cache{};

	FullyConnectedLayer () {
		// Kaiming He initialisation

		mt19937 rng(chrono::high_resolution_clock::now().time_since_epoch().count());
		constexpr double stddev = sqrt(2.0 / (N * N * in_channels));
		std::normal_distribution<double> gen{0, stddev};

		for (int i = 0; i < M * M * out_channels; i++)
			for (int j = 0; j < N * N * in_channels; j++)
				W[i][j] = gen(rng);
	}

	auto forward (const image<N, in_channels>& X) {
		auto arr_X {array_converted(X)};
		array<double, M * M * out_channels> arr_Y{};

		for (int i = 0; i < M * M * out_channels; i++)
			for (int j = 0; j < N * N * in_channels; j++)
				arr_Y[i] += arr_X[j] * W[i][j];

		return std::move(imagify<M, out_channels, M * M * out_channels>(arr_Y));
	}

	auto train (const image<N, in_channels>& X) {
			auto start = chrono::high_resolution_clock::now();
		
		copy_to_vector(X, cache);

		auto Y {forward(X)};

			auto stop = chrono::high_resolution_clock::now();

			cout << "fcon " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << '\n';

		return std::move(Y);
	}

	auto backward (const image<M, out_channels>& grad_Y) {

		auto arr_grad_Y {array_converted(grad_Y)};
		auto arr_last_X {array_converted(imagify<N, in_channels>(cache))};
		array<double, N * N * in_channels> arr_grad_X {};
		decltype(W) grad_W {};

		static array<array<Optimizer, N * N * in_channels>, M * M * out_channels> optimizer;

		for (int i = 0; i < M * M * out_channels; i++)
			for (int j = 0; j < N * N * in_channels; j++)
				arr_grad_X[j] += arr_grad_Y[i] * W[i][j],
				grad_W[i][j] = arr_grad_Y[i] * arr_last_X[j],
				optimizer[i][j].optimize(W[i][j], grad_W[i][j]);

		return std::move(imagify<N, in_channels, N*N*in_channels>(arr_grad_X));
	}

	void save (const string path) {
		ofstream out(path);
		out << fixed << setprecision(10);

		for (int i = 0; i < M * M * out_channels; i++)
			for (int j = 0; j < N * N * in_channels; j++)
				out << W[i][j] << ",";

	}
};
