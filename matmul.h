#ifndef NN_OPERATIONS_FastMatMul_H_
#define NN_OPERATIONS_FastMatMul_H_

// namespace nn {
namespace operations {

struct FastMatMul {
	static constexpr int MAX_WIDTH = 2048;
	static double buf0 alignas(64) [MAX_WIDTH];
	static double buf1 alignas(64) [MAX_WIDTH];
	static double buf2 alignas(64) [MAX_WIDTH];
	static double buf3 alignas(64) [MAX_WIDTH];

	template<int N, int M>
	using mat = array<array<double, M>, N>;

	template<uint64_t N, uint64_t M, uint64_t K>
	static auto fast_mat_mul (const mat<N, M>& A, const mat<M, K>& B, mat<N, K>& C) {

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

};

double FastMatMul::buf0[];
double FastMatMul::buf1[];
double FastMatMul::buf2[];
double FastMatMul::buf3[];

} // namespace operations
// } // namespace nn

#endif // NN_OPERATIONS_FastMatMul_H_