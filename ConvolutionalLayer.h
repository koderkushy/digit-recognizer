template<
	class Optimizer,
	int in_channels,
	int out_channels,
	int K,
	int P = 0
>
struct ConvolutionalLayer {

	array<image<K, in_channels>, out_channels> W{};
	vector<double> cache{};

	ConvolutionalLayer () {
		// Kaiming He initialisation

		mt19937 rng(chrono::high_resolution_clock::now().time_since_epoch().count());
		constexpr double stddev = sqrt(2.0 / (K * K * in_channels));
		std::normal_distribution<double> N{0, stddev};

		for (int o = 0; o < out_channels; o++)
			for (int i = 0; i < in_channels; i++)
				for (int x = 0; x < K; x++)
					for (int y = 0; y < K; y++)
						W[o][i][x][y] = N(rng);

	}

	template<uint64_t N>
	auto forward (const image<N, in_channels>& X_unpadded) {
		static constexpr int M = N + (P * 2) - K + 1;

		return convolve(pad<N, in_channels, P>(X_unpadded), W);
	}

	template<uint64_t N>
	auto train (const image<N, in_channels>& X) {
			auto start = chrono::high_resolution_clock::now();

		copy_to_vector(X, cache);
		auto Y {forward(X)};

			auto stop = chrono::high_resolution_clock::now();

			cout << "conv " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << '\n';

		return std::move(Y);

	}

	template<uint64_t M>
	auto backward (const image<M, out_channels>& grad_Y) {
		static constexpr int N = M - P * 2 + K - 1;

		array<image<K, in_channels>, out_channels> grad_W {};
		auto last_X {imagify<N, in_channels>(cache)};

		for (int f = 0; f < out_channels; f++)
			for (int g = 0; g < in_channels; g++)
				grad_W[f][g] = convolve(pad<N, P>(last_X[g]), grad_Y[f])[0];

		image<N, in_channels> grad_X {};
		auto grad_Y_padded {pad<M, out_channels, K - 1>(grad_Y)};
		decltype(W) W_flipped {};

		for (int f = 0; f < out_channels; f++)
			for (int g = 0; g < in_channels; g++)
				for (int i = 0; i < K; i++)
					for (int j = 0; j < K; j++)
						W_flipped[f][g][i][j] = W[f][g][K - i - 1][K - j - 1];

		for (int g = 0; g < in_channels; g++) {
			for (int f = 0; f < out_channels; f++) {
				auto component {convolve(grad_Y_padded[f], W_flipped[f][g])[0]};
				for (int i = 0; i < N; i++)
					for (int j = 0; j < N; j++)
						grad_X[g][i][j] += component[i + P][j + P];
			}
		}

		static array<array<array<array<Optimizer, K>, K>, in_channels>, out_channels> optimizer{};
		for (int o = 0; o < out_channels; o++)
			for (int i = 0; i < in_channels; i++)
				for (int x = 0; x < K; x++)
					for (int y = 0; y < K; y++)
						optimizer[o][i][x][y].optimize(W[o][i][x][y], grad_W[o][i][x][y]);

		// static array<Optimizer<K, in_channels>, out_channels> optimizer;
		// for (int f = 0; f < out_channels; f++)
		// 	optimizer[f].optimize(W[f], grad_W[f]);

		return std::move(grad_X);
	}

	void save (const string path) {
		ofstream out(path);
		out << fixed << setprecision(10);

		for (int o = 0; o < out_channels; o++)
			for (int i = 0; i < in_channels; i++)
				for (int x = 0; x < K; x++)
					for (int y = 0; y < K; y++)
						out << W[o][i][x][y] << ",";
	}
};
