struct DropOut {
	mt19937 rng;
	static constexpr uint64_t R = std::numeric_limits<uint32_t>::max();

	DropOut (): rng(chrono::high_resolution_clock::now().time_since_epoch().count()) {}

	vector<vector<vector<bool>>> memo;

	template<uint64_t N, uint64_t channels>
	auto evaluate (image<N, channels> X, const double p) {
		memo = vector(channels, vector(N, vector(N, false)));

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (rng() < R * p)
						memo[f][i][j] = true, X[f][i][j] = 0;

		return std::move(X);
	}

	template<int N, int channels>
	auto back_propagate (image<N, channels> grad_Y) {

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (memo[f][i][j])
						grad_Y[f][i][j] = 0;

		return std::move(grad_Y);
	}
};