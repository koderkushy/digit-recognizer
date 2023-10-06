#ifndef NN_OPERATIONS_FASTMATH_H_
#define NN_OPERATIONS_FASTMATH_H_

using uint = long unsigned int;

namespace nn {
namespace math {

struct FastMath {
	static constexpr uint MAX_WIDTH = 2048;
	static float buf0 alignas(64) [MAX_WIDTH];
	static float buf1 alignas(64) [MAX_WIDTH];
	static float buf2 alignas(64) [MAX_WIDTH];
	static float buf3 alignas(64) [MAX_WIDTH];

	template<uint N, uint M>
	using mat = std::array<std::array<float, M>, N>;

	template<uint N, uint M, uint K>
	static auto fast_mat_mul (const mat<N, M>& A, const mat<M, K>& B, mat<N, K>& C) {

		static_assert(K <= MAX_WIDTH);

		int j = 0;
		for ( ; j + 4 <= N; j += 4) {
			std::fill(buf0, buf0 + K, 0);
			std::fill(buf1, buf1 + K, 0);
			std::fill(buf2, buf2 + K, 0);
			std::fill(buf3, buf3 + K, 0);

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
			std::copy(buf0, buf0 + K, C[j + 0].begin());
			std::copy(buf1, buf1 + K, C[j + 1].begin());
			std::copy(buf2, buf2 + K, C[j + 2].begin());
			std::copy(buf3, buf3 + K, C[j + 3].begin());
		}

		for ( ; j < N; ++j) {
			std::fill(buf0, buf0 + K, 0);
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

			std::copy(buf0, buf0 + K, C[j].begin());
		}
		
	}

	template<uint N, uint M, uint K>
	static auto mat_mul (const std::array<std::array<float, N>, M>& A, const std::array<std::array<float, K>, N>& B) {
		std::array<std::array<float, K>, M> C {};

		if constexpr (std::min({N, M, K}) > 50)
			fast_mat_mul(A, B, C);
		else
			for (int i = 0; i < M; i++)
					for (int k = 0; k < N; k++)
#pragma GCC ivdep
				for (int j = 0; j < K; j++)
						C[i][j] += A[i][k] * B[k][j];

		return std::move(C);
	}
};

float FastMath::buf0[];
float FastMath::buf1[];
float FastMath::buf2[];
float FastMath::buf3[];

} // namespace math
} // namespace nn

#endif // NN_OPERATIONS_FASTMATH_H_