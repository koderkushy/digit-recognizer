template<
	template<int N, int channels> class Optimizer,
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

	template<int N, int C>
	auto pad (const image<N, C>& a, const int k) const {
		if (k == 0) return a;
		image<N + k * 2, C> b{};
		for (int f = 0; f < C; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					b[f][i + k][j + k] = a[f][i][j];
		return std::move(b);
	}

	template<uint64_t N>
	auto forward (const image<N, in_channels>& X) {
		static constexpr int M = N + (P * 2) - K + 1;
		image<M, out_channels> Y{};

		for (int f = 0; f < out_channels; f++)
			for (int z = 0; z < in_channels; z++)
				for (int i = 0; i < M; i++)
					for (int j = 0; j < M; j++)
						for (int x = max(0, P - i); x < min(K, N - i + P); x++)
							for (int y = max(0, P - j); y < min(K, N - j + P); y++)
								Y[f][i][j] += X[z][i + x - P][j + y - P] * W[f][z][x][y];

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

		array<image<K, in_channels>, out_channels> grad_W{};
		auto last_X = imagify<N, in_channels>(cache);

		for (int f = 0; f < out_channels; f++)
			for (int z = 0; z < in_channels; z++)
				for (int i = 0; i < M; i++)
					for (int j = 0; j < M; j++)
						for (int x = max(0, P - i); x < min(K, N - i + P); x++)
							for (int y = max(0, P - j); y < min(K, N - j + P); y++)
								grad_W[f][z][x][y] += last_X[z][i + x - P][j + y - P] * grad_Y[f][i][j];

		// for (int f = 0; f < out_channels; f++)
		// 	for (int i = 0; i < in_channels; i++)
		// 		for (int j = 0; j < K; j++)
		// 			for (int k = 0; k < K; k++)
		// 				for (int x = 0; x < M; x++)
		// 					for (int y = 0; y < M; y++)
		// 						grad_W[f][i][j][k] += last_X[i][j][k] * grad_Y[f][x][y];

		image<N, in_channels> grad_X{};

		for (int f = 0; f < out_channels; f++)
			for (int z = 0; z < in_channels; z++)
				for (int i = 0; i < N; i++)
					for (int j = 0; j < N; j++)
						for (int x = min(K - 1, i + P); x > max(-1, K - 1 - N + i - P); x--)
							for (int y = min(K - 1, j + P); y > max(-1, K - 1 - N + j - P); y--)
								grad_X[z][i][j] += W[f][z][x][y] * grad_Y[f][i + P - x][j + P - y];

		// for (int f = 0; f < in_channels; f++)
		// 	for (int i = 0; i < N; i++)
		// 		for (int j = 0; j < N; j++)
		// 			for (int k = 0; k < out_channels; k++)
		// 				for (int x = min(K - 1, i); x > max(-1, K - 1 - N + i); x--)
		// 					for (int y = min(K - 1, j); y > max(-1, K - 1 - N + j); y--)
		// 						grad_X[f][i][j] += W[k][f][x][y] * grad_Y[k][i - x][j - y];

		static array<Optimizer<K, in_channels>, out_channels> optimizer;
		for (int f = 0; f < out_channels; f++)
			optimizer[f].optimize(W[f], grad_W[f]);
		// optimizer.optimize(W, grad_W);

		return std::move(grad_X);
	}
};
