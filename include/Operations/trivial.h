#ifndef NN_OPERATIONS_TRIVIAL_H_
#define NN_OPERATIONS_TRIVIAL_H_

namespace nn {
namespace operations {

	template<int N> using filter = array<array<double, N>, N>;
	template<int N, int C> using image = array<filter<N>, C>;
	template<typename T, typename U> T min (const T& x, const U& y) { return std::min(x, static_cast<T>(y)); }
	template<typename T, typename U> T max (const T& x, const U& y) { return std::max(x, static_cast<T>(y)); }

	template<uint64_t N, uint64_t channels>
	auto copy_to_vector (const image<N, channels>& X, vector<double>& V) {
		V.clear(), V.reserve(N * N * channels);

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					V.emplace_back(X[f][i][j]);
	}

	template<uint64_t N, uint64_t channels>
	auto array_converted (const image<N, channels>& X) {
		array<double, N * N * channels> Y{};
		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					Y[(f * N + i) * N + j] = X[f][i][j];
		return Y;
	}

	template<uint64_t N, uint64_t channels, uint64_t W>
	auto imagify (const array<double, W>& X) {
		static_assert(W == N * N * channels);
		image<N, channels> Y{};

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					Y[f][i][j] = X[(f * N + i) * N + j];

		return Y;
	}

	template<uint64_t N, uint64_t channels>
	auto imagify (const vector<double>& V) {
		assert(N * N * channels == V.size());
		image<N, channels> X{};

		for (int i = 0; i < N * N * channels; i++)
			X[i / (N * N)][(i / N) % N][i % N] = V[i];

		return X;
	}

	template<int N, int C, int P>
	auto pad (const image<N, C>& X, const double val = 0) {
		image<N + P * 2, C> Y{};
		if constexpr (P == 0)
			Y = X;
		else if constexpr (P > 0) {
			for (int c = 0; c < C; c++) {
				for (int i = 0; i < N; i++)
					for (int j = 0; j < N; j++)
						Y[c][i + P][j + P] = X[c][i][j];
				for (int i = 0; i < P; i++)
					for (int j = 0; j < P; j++)
						Y[c][i][j] = Y[c][N - i - 1][j] = Y[c][i][N - j - 1] = Y[c][N - i - 1][N - j - 1] = val;
			}
		} else {
			for (int c = 0; c < C; c++)
				for (int i = P; i < N - P; i++)
					for (int j = P; j < N - P; j++)
						Y[c][i - P][j - P] = X[c][i][j];
		}
		return std::move(Y);
	}

	template<int N, int P>
	auto pad (const filter<N>& X, const double val = 0) {
		filter<N + P * 2> Y{};
		if constexpr (P == 0)
			Y = X;
		else if constexpr (P > 0) {
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					Y[i + P][j + P] = X[i][j];
			for (int i = 0; i < P; i++)
				for (int j = 0; j < P; j++)
					Y[i][j] = Y[N - i - 1][j] = Y[i][N - j - 1] = Y[N - i - 1][N - j - 1] = val;
		} else {
			for (int i = P; i < N - P; i++)
				for (int j = P; j < N - P; j++)
					Y[i - P][j - P] = X[i][j];
		}
		return std::move(Y);
	}

} // namespace operations
} // namespace nn

#endif // NN_OPERATIONS_TRIVIAL_H_