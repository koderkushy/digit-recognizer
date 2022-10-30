#ifndef NN_OPERATIONS_MATH_H_
#define NN_OPERATIONS_MATH_H_

#include "trivial.h"

namespace nn {
namespace operations {

struct Math {
	static constexpr int MAX_WIDTH = 2048;
	static double buf0 alignas(64) [MAX_WIDTH];
	static double buf1 alignas(64) [MAX_WIDTH];
	static double buf2 alignas(64) [MAX_WIDTH];
	static double buf3 alignas(64) [MAX_WIDTH];

	template<int N, int M>
	using mat = array<array<double, M>, N>;

	template<uint64_t N, uint64_t M, uint64_t K>
	static auto mat_mul (const mat<N, M>& A, const mat<M, K>& B, mat<N, K>& C) {

		if constexpr (std::min({ N, M, K }) < 20) {
			for (int i = 0; i < M; i++)
				for (int k = 0; k < N; k++) {
					const auto& a = A[i][k];
					auto& c = C[i];
					const auto& b = B[k];
 			#pragma GCC ivdep
					for (int j = 0; j < K; j++)
						c[j] += a * b[j];
				}

		} else {

			static_assert(K <= MAX_WIDTH);

			int j = 0;
			for ( ; j + 4 <= N; j += 4) {
				fill(buf0, buf0 + K, 0);
				fill(buf1, buf1 + K, 0);
				fill(buf2, buf2 + K, 0);
				fill(buf3, buf3 + K, 0);

				int k = 0;
				for ( ; k + 25 <= M; k += 25) {
					for (int _k = k; _k < k + 25; _k++) {
						const auto& a0 = A[j + 0][_k];
						const auto& a1 = A[j + 1][_k];
						const auto& a2 = A[j + 2][_k];
						const auto& a3 = A[j + 3][_k];
						const auto& b = B[_k];
		#pragma GCC ivdep
						for (int i = 0; i < K; i++) {
							const auto& x = b[i];
							buf0[i] += a0 * x;
							buf1[i] += a1 * x;
							buf2[i] += a2 * x;
							buf3[i] += a3 * x;
						}
					}
				}
				for ( ; k < M; ++k) {
					const auto& a0 = A[j + 0][k];
					const auto& a1 = A[j + 1][k];
					const auto& a2 = A[j + 2][k];
					const auto& a3 = A[j + 3][k];
					const auto& b = B[k];
		#pragma GCC ivdep
					for (int i = 0; i < K; i++) {
						const auto& x = b[i];
						buf0[i] += a0 * x;
						buf1[i] += a1 * x;
						buf2[i] += a2 * x;
						buf3[i] += a3 * x;
					}
				}
				copy(buf0, buf0 + K, C[j + 0].begin());
				copy(buf1, buf1 + K, C[j + 1].begin());
				copy(buf2, buf2 + K, C[j + 2].begin());
				copy(buf3, buf3 + K, C[j + 3].begin());
			}

			for ( ; j < N; ++j) {
				fill(buf0, buf0 + K, 0);
				int k = 0;
				for ( ; k + 25 <= M; ++k) {
					for (int _k = k; _k < k + 25; ++_k) {
						const auto& a = A[j][_k];
						const auto& b = B[_k];
						for (int i = 0; i < K; i++)
							buf0[i] += a * b[i];
					}
				}

				for ( ; k < M; ++k) {
					const auto& a = A[j][k];
					const auto& b = B[k];
					for (int i = 0; i < K; i++)
							buf0[i] += a * b[i];
				}

				copy(buf0, buf0 + K, C[j].begin());
			}
		}
	}

	template<uint64_t M, uint64_t N, uint64_t K>
	static auto mat_mul (const array<array<double, N>, M>& A, const array<array<double, K>, N>& B) {
		array<array<double, K>, M> C {};

		if constexpr (std::min({N, M, K}) > 50)
			Math::mat_mul(A, B, C);
		else
			for (int i = 0; i < M; i++)
					for (int k = 0; k < N; k++)
 #pragma GCC ivdep
				for (int j = 0; j < K; j++)
						C[i][j] += A[i][k] * B[k][j];

		return std::move(C);
	}

	template<uint64_t N, uint64_t NC, uint64_t K, uint64_t KC>
	static auto convolve (const image<N, NC>& X, const array<image<K, NC>, KC>& W) {
		static constexpr int M = N - K + 1;
		image<M, KC> Y{};

		array<array<double, K * K * NC>, KC> W_mat{};
		for (int f = 0; f < KC; f++)
			// W_mat[f] = array_converted(W[f]);
			for (int g = 0; g < NC; g++)
				for (int i = 0; i < K; i++)
					for (int j = 0; j < K; j++)
						W_mat[f][(g * K + i) * K + j] = W[f][g][i][j];

		array<array<double, M * M>, K * K * NC> X_mat{};	
		for (int i = 0; i < M; i++)
			for (int j = 0; j < M; j++)
				for (int g = 0; g < NC; g++)
					for (int x = 0; x < K; x++)
						for (int y = 0; y < K; y++)
							X_mat[y + K * (x + K * g)][i * M + j] = X[g][i + x][j + y];

		// array<array<double, M * M>, KC> Y_mat { };
		// Math::mat_mul(W_mat, X_mat, Y_mat);

		array<array<double, M * M>, KC> Y_mat { Math::mat_mul(W_mat, X_mat)};

		for (int f = 0; f < KC; f++)
			Y[f] = imagify<M, 1, M * M>(Y_mat[f])[0];

		return std::move(Y);
	}


	template<uint64_t N, uint64_t K>
	static auto convolve (const filter<N>& X, const filter<K>& W) {
		image<N, 1> _X{}; _X[0] = X;
		array<image<K, 1>, 1> _W{}; _W[0][0] = W;
		return convolve(_X, _W);
	}

};

double Math::buf0[];
double Math::buf1[];
double Math::buf2[];
double Math::buf3[];

} // namespace operations
} // namespace nn

#endif // NN_OPERATIONS_MATH_H_