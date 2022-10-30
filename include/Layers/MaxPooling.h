#ifndef NN_MAX_POOLING_H_
#define NN_MAX_POOLING_H_

#include "../Operations/trivial.h"

namespace nn {

template<
	int K,
	int P,
	int S,
	class NextLayer
>
struct MaxPool {
	static_assert(P >= 0 and K > P and S > 0);

	NextLayer L { };

	MaxPool () {}

	template<uint64_t _N, uint64_t channels>
	auto recurse (const nn::operations::image<_N, channels>& _X, const int label) {
		static constexpr int N = _N + 2 * P;
		static constexpr int M = (N - K + S) / S;

		auto X { nn::operations::pad<_N, channels, P>(_X, std::numeric_limits<double>::min()) };

		auto [gradient, loss] = L.recurse(forward(X), label);
		return pair(backward(X, gradient), loss);
	}
	
	template<uint64_t _N, uint64_t channels>
	auto evaluate (const nn::operations::image<_N, channels>& _X, const int label) {

		auto X { nn::operations::pad<_N, channels, P>(_X, std::numeric_limits<double>::min()) };
		return L.evaluate(forward(X), label);
	}

	template<uint64_t _N, uint64_t channels>
	auto forward (const nn::operations::image<_N, channels>& _X) {
		static constexpr int N = _N + 2 * P;
		static constexpr int M = (N - K + S) / S;

		auto X { nn::operations::pad<_N, channels, P>(_X, std::numeric_limits<double>::min()) };

		nn::operations::image<M, channels> Y { };

		for (int c = 0; c < channels; c++)
			for (int i = 0; i + K <= N; i += S)
				for (int j = 0; j + K <= N; j += S) {
					Y[c][i / S][j / S] = X[c][i][j];

					for (int x = 0; x < K; x++)
						for (int y = 0; y < K; y++)
							Y[c][i / S][j / S] = std::max(Y[c][i / S][j / S], X[c][i + x][j + y]);
				}

		return std::move(Y);
	}

	auto optimize () {
		L.optimize();
	}

	template<uint64_t N, uint64_t M, uint64_t channels>
	auto backward (const nn::operations::image<N, channels>& X, const nn::operations::image<M, channels>& grad_Y) {
		static_assert(M == (N - K + S) / S);

		nn::operations::image<N, channels> grad_X { };

		for (int c = 0; c < channels; c++)
			for (int i = 0; i + K <= N; i += S)
				for (int j = 0; j + K <= N; j += S) {
					auto v = X[c][i][j];

					for (int x = 0; x < K; x++)
						for (int y = 0; y < K; y++)
							v = std::max(v, X[c][i + x][j + y]);
					for (int x = 0; x < K; x++)
						for (int y = 0; y < K; y++)
							if (v - X[c][i + x][j + y] < 1e-7)
								grad_X[c][i + x][j + y] += grad_Y[c][i / S][j / S];
				}

		return nn::operations::pad<N, channels, -P>(grad_X);
	}

	auto save (const string path) {
		L.save(path);
	}
};

} // namespace nn

#endif // NN_MAX_POOLING_H_