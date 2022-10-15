#include "bits/stdc++.h"
using namespace std;

constexpr double eta = 0.1;

template<int N>
using filter = array<array<double, N>, N>;

template<int N, int C>
using image = array<filter<N>, C>;

template<int in_channels, int out_channels, int K, int P = 0>
struct CNNLayer {

	array<image<K, in_channels>, out_channels> W;

	template<typename T, typename U>
	T min (const T& x, const U& y) { return std::min(x, static_cast<T>(y)); }
	template<typename T, typename U>
	T max (const T& x, const U& y) { return std::max(x, static_cast<T>(y)); }

	template<int N, int C>
	auto pad (const image<N, C>& a, const int k) const {
		if (k == 0) return a;
		image<N + k * 2, C> b{};
		for (int f = 0; f < C; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					b[f][i + k][j + k] = a[f][i][j];
		return b;
	}

	template<int N, int M, int C, int pad = 0>
	auto conv (const image<N, C>& a, const image<M, C>& b) {
		constexpr int S = N + 2 * pad - M + 1;
		filter<S> res{};

		for (int i = 0; i < S; i++)
			for (int j = 0; j < S; j++)
				for (int x = max(0, pad - i); x < min(M, N - i + pad); x++)
					for (int y = max(0, pad - j); y < min(M, N - j + pad); y++)
						for (int z = 0; z < C; z++)
							res[i][j] += a[i + x - pad][j + y - pad][z] * b[x][y][z];

		return res;
	}

	template<uint64_t N>
	auto evaluate (const image<N, in_channels>& X) {
		static constexpr int M = N + (P * 2) - K + 1;
		image<M, out_channels> Y{};

		for (int f = 0; f < out_channels; f++)
			Y[f] = conv(X, W[f]);

		return Y;
	}

	template<int N, int M>
	auto back_propagate (const image<N, in_channels>& X, const image<M, out_channels>& grad_Y) {
		assert(M == N - K + 1);

		array<image<K, in_channels>, out_channels> grad_W{};

		for (int f = 0; f < out_channels; f++)
			for (int i = 0; i < in_channels; i++)
				for (int j = 0; j < K; j++)
					for (int k = 0; k < K; k++)
						for (int x = 0; x < M; x++)
							for (int y = 0; y < M; y++)
								grad_W[f][i][j][k] += X[i][j + x][k + y] * grad_Y[f][x][y];

		image<N, in_channels> grad_X{};

		for (int f = 0; f < in_channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					for (int k = 0; k < out_channels; k++)
						for (int x = min(K - 1, i); x > max(-1, K - 1 - N + i); x--)
							for (int y = min(K - 1, j); y > max(-1, K - 1 - N + j); y--)
								grad_X[f][i][j] += W[k][f][x][y] * grad_Y[k][i - x][j - y];

		for (int f = 0; f < out_channels; f++)
			for (int i = 0; i < in_channels; i++)
				for (int j = 0; j < K; j++)
					for (int k = 0; k < K; k++)
						W[f][i][j][k] += -eta * grad_W[f][i][j][k];
		
		return grad_X;
	}

};

template<int channels, int K, int P = 0, int stride = 1>
struct MaxPool {
	static_assert(stride > 0 and P >= 0 and K > P);

	template<int N>
	auto evaluate (const image<N, channels>& X) {
		static constexpr int M = (N + P * 2) / stride;
		image<M, channels> b{};

		for (int c = 0; c < channels; c++)
			for (int i = -P; i < N + P; i += stride)
				for (int j = -P; j < N + P; j += stride) {
					auto &v = b[(i + P) / stride][(j + P) / stride] = std::numeric_limits<double>::min();
					for (int x = max(0, -i); x < min(K, N - i); x++)
						for (int y = max(0, -j); y < min(K, N - j); y++)
							v = max(v, X[i + x][j + y]);
				}

		return b;
	}

	template<int N, int M>
	auto back_propagate (const image<N, channels>& X, const image<M, channels>& grad_Y) {
		static_assert(M == (N + P * 2) / stride);
		
	}
};

struct ReLu {
	
};

int main(){
    ios_base::sync_with_stdio(0), cin.tie(0);

    CNNLayer<1, 32, 5> conv1;
    CNNLayer<32, 32, 5> conv2;
    CNNLayer<32, 64, 3> conv3;
    CNNLayer<64, 64, 3> conv4;



}