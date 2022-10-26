struct DropOut {
	mt19937 rng;
	static constexpr uint64_t R = std::numeric_limits<uint32_t>::max();

	DropOut (): rng(chrono::high_resolution_clock::now().time_since_epoch().count()) {}

	vector<bool> cache{};

	template<uint64_t N, uint64_t channels>
	auto train (image<N, channels> X, const double p = 0.5) {

		assert(p > 0 and p < 1);
		cache.resize(N * N * channels);

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (rng() < R * p)
						cache[(f * N + i) * N + j] = true, X[f][i][j] = 0;

		return std::move(X);
	}

	template<uint64_t N, uint64_t channels>
	auto backward (image<N, channels> grad_Y) {
		assert(cache.size() == channels * N * N);

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (cache[(f * N + i) * N + j])
						grad_Y[f][i][j] = 0;

		cache.clear();
		return std::move(grad_Y);
	}
};