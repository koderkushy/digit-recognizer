template<
	class Optimizer,
	int in_channels,
	int out_channels,
	int K,
	int P = 0
>
struct ConvolutionalLayer {

	array<image<K, in_channels>, out_channels> W { };
	array<double, out_channels> b { };
	vector<double> cache { };

	ConvolutionalLayer () {
		// Kaiming He initialisation

		mt19937 rng(chrono::high_resolution_clock::now().time_since_epoch().count());
		std::normal_distribution<double> 
			weight_gaussian { 0, sqrt(2.0 / K * K * in_channels) },
			bias_gaussian { 0, sqrt(2.0 / 10) };

		for (int o = 0; o < out_channels; o++)
			for (int i = 0; i < in_channels; i++)
				for (int x = 0; x < K; x++)
					for (int y = 0; y < K; y++)
						W[o][i][x][y] = weight_gaussian(rng);

		for (int f = 0; f < out_channels; f++)
			b[f] = bias_gaussian(rng);

	}

	template<uint64_t N>
	auto forward (const image<N, in_channels>& X_unpadded) {
		static constexpr int M = N + (P * 2) - K + 1;

		auto Y { convolve(pad<N, in_channels, P>(X_unpadded), W) };
		for (int f = 0; f < out_channels; f++)
			for (int i = 0; i < M; i++)
				for (int j = 0; j < M; j++)
					Y[f][i][j] += b[f];

		return std::move(Y);
	}

	template<uint64_t N>
	auto train (const image<N, in_channels>& X) {
		copy_to_vector(X, cache);
		return forward(X);
	}

	template<uint64_t M>
	auto backward (const image<M, out_channels>& grad_Y) {
		static constexpr int N = M - P * 2 + K - 1;

		static array<array<array<array<Optimizer, K>, K>, in_channels>, out_channels> W_optimizer{};
		static array<Optimizer, out_channels> b_optimizer { };

		auto X { imagify<N, in_channels>(cache) };
		auto W_flipped { W };

		decltype(W) grad_W { };
		decltype(b) grad_b { };
		decltype(X) grad_X { };

		// Compute gradients wrt W
		for (int g = 0; g < in_channels; g++) {
			auto X_g { pad<N, P>(X[g]) };

			for (int f = 0; f < out_channels; f++)
				grad_W[f][g] = convolve(X_g, grad_Y[f])[0];
		}

		// Flip W
		for (int f = 0; f < out_channels; f++)
			for (int g = 0; g < in_channels; g++)
				for (int i = 0; i < K; i++) {
					for (int j = 0; j < K; j++)
						reverse(W_flipped[f][g][i].begin(), W_flipped[f][g][i].end());
					reverse(W_flipped[f][g].begin(), W_flipped[f][g].end());
				}

		// Compute gradients wrt X
		auto grad_Y_padded { pad<M, out_channels, K - 1>(grad_Y) };

		for (int g = 0; g < in_channels; g++) {
			for (int f = 0; f < out_channels; f++) {
				auto component {convolve(grad_Y_padded[f], W_flipped[f][g])[0]};

				for (int i = 0; i < N; i++)
					for (int j = 0; j < N; j++)
						grad_X[g][i][j] += component[i + P][j + P];
			}
		}

		// Compute gradients wrt b
		for (int f = 0; f < out_channels; f++)
			for (int i = 0; i < M; i++)
				for (int j = 0; j < M; j++)
					grad_b[f] += grad_Y[f][i][j];

		// Optimize W
		for (int o = 0; o < out_channels; o++)
			for (int i = 0; i < in_channels; i++)
				for (int x = 0; x < K; x++)
					for (int y = 0; y < K; y++)
						W_optimizer[o][i][x][y].optimize(W[o][i][x][y], grad_W[o][i][x][y]);

		// Optimize b
		for (int f = 0; f < out_channels; f++)
			b_optimizer[f].optimize(b[f], grad_b[f]);


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
