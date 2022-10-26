template<
	int N,
	int K,
	int P = 0,
	int S = 1
>
struct MaxPool {
	static_assert(P >= 0 and K > P and S > 0);
	static constexpr int M = (N + 2 * P - K + S) / S;
	vector<double> cache;

	MaxPool () {}

	template<uint64_t channels>
	auto forward (const image<N, channels>& X) {
		image<M, channels> Y{};

		for (int c = 0; c < channels; c++)
			for (int i = -P; i < N + P - K + 1; i += S)
				for (int j = -P; j < N + P - K + 1; j += S) {
					auto &v
						= Y[c][(i + P) / S][(j + P) / S]
							= std::numeric_limits<double>::min();

					for (int x = max(0, -i); x < min(K, N - i); x++)
						for (int y = max(0, -j); y < min(K, N - j); y++)
							v = max(v, X[c][i + x][j + y]);
				}

		return std::move(Y);
	}

	template<uint64_t channels>
	auto train (const image<N, channels>& X) {
		copy_to_vector(X, cache);
		return forward(X);
	}

	template<uint64_t channels>
	auto backward (const image<M, channels>& grad_Y) {
		image<N, channels> grad_X{};
		auto last_X{imagify<N, channels>(cache)};

		for (int c = 0; c < channels; c++)
			for (int i = -P; i < N + P - K + 1; i += S)
				for (int j = -P; j < N + P - K + 1; j += S) {	
					auto max_value = std::numeric_limits<double>::min();
					for (int x = max(0, -i); x < min(K, N - i); x++)
						for (int y = max(0, -j); y < min(K, N - j); y++)
							if (max_value < last_X[c][i + x][j + y])
								max_value = last_X[c][i + x][j + y];
					for (int x = max(0, -i); x < min(K, N - i); x++)
						for (int y = max(0, -j); y < min(K, N - j); y++)
							if (last_X[c][i + x][j + y] == max_value)
								grad_X[c][i + x][j + y] += grad_Y[c][(i + P) / S][(j + P) / S];
				}

		return std::move(grad_X);
	}
};
