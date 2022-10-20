template<uint64_t K, int P = 0>
struct MaxPool {
	static_assert(P >= 0 and K > P);

	vector<vector<vector<double>>> last_X;

	template<uint64_t N, uint64_t channels>
	auto forward (const image<N, channels>& X) {
		static constexpr int M = N + P * 2 - K + 1;
		image<M, channels> Y{};

		for (int c = 0; c < channels; c++)
			for (int i = -P; i < N + P; i++)
				for (int j = -P; j < N + P; j++) {
					auto &v
						= Y[c][i + P][j + P]
							= std::numeric_limits<double>::min();

					for (int x = max(0, -i); x < min(K, N - i); x++)
						for (int y = max(0, -j); y < min(K, N - j); y++)
							v = max(v, X[c][i + x][j + y]);
				}

		return std::move(Y);
	}

	template<uint64_t N, uint64_t channels>
	auto train (const image<N, channels>& X) {
		copy_to_vector(X, last_X);
		return forward(X);
	}

	template<uint64_t M, uint64_t channels>
	auto backward (const image<M, channels>& grad_Y) {
		static constexpr int N = M + K - 1 - P * 2;

		image<N, channels> grad_X{};

		for (int c = 0; c < channels; c++)
			for (int i = -P; i < N + P; i++)
				for (int j = -P; j < N + P; j++) {
					auto max_value = std::numeric_limits<double>::min();
					for (int x = max(0, -i); x < min(K, N - i); x++)
						for (int y = max(0, -j); y < min(K, N - j); y++)
							if (max_value < last_X[c][i + x][j + y])
								max_value = last_X[c][i + x][j + y];
					for (int x = max(0, -i); x < min(K, N - i); x++)
						for (int y = max(0, -j); y < min(K, N - j); y++)
							if (last_X[c][i + x][j + y] == max_value)
								grad_X[c][i + x][j + y] += grad_Y[c][(i + P)][(j + P)];
				}

		return std::move(grad_X);
	}
};