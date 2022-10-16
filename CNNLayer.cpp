#include "bits/stdc++.h"
using namespace std;

constexpr double eta = 0.1;

template<int N>
using filter = array<array<double, N>, N>;

template<int N, int C>
using image = array<filter<N>, C>;

template<typename T, typename U>
T min (const T& x, const U& y) { return std::min(x, static_cast<T>(y)); }
template<typename T, typename U>
T max (const T& x, const U& y) { return std::max(x, static_cast<T>(y)); }

template<int N, int channels>
auto copy_to_vector (const image<N, channels>& X, vector<vector<vector<double>>>& V) {
	V = vector(channels, vector(N, vector(N, double())));

	for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					V[f][i][j] = X[f][i][j];
}

template<int in_channels, int out_channels, int K, int P = 0>
struct ConvolutionalLayer {

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
		return std::move(b);
	}

	vector<vector<vector<double>>> last_X;

	template<uint64_t N>
	auto evaluate (const image<N, in_channels>& X) {
		static constexpr int M = N + (P * 2) - K + 1;
		image<M, out_channels> Y{};

		for (int f = 0; f < out_channels; f++)
			for (int i = 0; i < M; i++)
				for (int j = 0; j < M; j++)
					for (int x = max(0, P - i); x < min(M, N - i + P); x++)
						for (int y = max(0, P - j); y < min(M, N - j + P); y++)
							for (int z = 0; z < in_channels; z++)
								Y[f][i][j] += X[z][i + x - P][j + y - P] * W[f][z][x][y];

		return std::move(Y);
	}

	template<uint64_t N>
	auto train (const image<N, in_channels>& X) {
		copy_to_vector(X, last_X);
		return evaluate(X);
	}

	template<uint64_t M>
	auto back_propagate (const image<M, out_channels>& grad_Y) {
		static constexpr int N = M + K - 1;

		array<image<K, in_channels>, out_channels> grad_W{};

		for (int f = 0; f < out_channels; f++)
			for (int i = 0; i < in_channels; i++)
				for (int j = 0; j < K; j++)
					for (int k = 0; k < K; k++)
						for (int x = 0; x < M; x++)
							for (int y = 0; y < M; y++)
								grad_W[f][i][j][k] += last_X[i][j + x][k + y] * grad_Y[f][x][y];

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
		
		return std::move(grad_X);
	}

};

template<uint64_t K, int P = 0>
struct MaxPool {
	static_assert(P >= 0 and K > P);

	vector<vector<vector<double>>> last_X;

	template<uint64_t N, uint64_t channels>
	auto evaluate (const image<N, channels>& X) {
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
		return evaluate(X);
	}

	template<uint64_t M, uint64_t channels>
	auto back_propagate (const image<M, channels>& grad_Y) {
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

struct ReLu {
	// y = max(0, x);

	vector<vector<vector<double>>> last_X;

	template<uint64_t N, uint64_t channels>
	auto evaluate (image<N, channels> X) {

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (X[f][i][j] < 0)
						X[f][i][j] = 0;

		return std::move(X);
	}

	template<int N, int channels>
	auto train (image<N, channels> X) {
		copy_to_vector(X, last_X);
		return evaluate(std::move(X));
	}

	template<uint64_t N, uint64_t channels>
	auto back_propagate (const image<N, channels>& grad_Y) {
		image<N, channels> grad_X{};

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (last_X[f][i][j] > 0)
						grad_X[f][i][j] = grad_Y[f][i][j];

		return std::move(grad_X);
	}

};

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

template<int N, int in_channels, int M, int out_channels>
struct FullyConnectedLayer {
	array<array<array<image<N, in_channels>, M>, M>, out_channels> W;

	vector<vector<vector<double>>> last_X;

	auto evaluate (const image<N, in_channels>& X) {
		image<M, out_channels> Y{};
		for (int f = 0; f < out_channels; f++)
			for (int i = 0; i < M; i++)
				for (int j = 0; j < M; j++)
					for (int x = 0; x < in_channels; x++)
						for (int y = 0; y < N; y++)
							for (int z = 0; z < N; z++)
								Y[f][i][j] += W[f][i][j][x][y][z] * X[x][y][z];
		return std::move(Y);
	}

	auto train (const image<N, in_channels>& X) {
		copy_to_vector(X, last_X);
		return evaluate(X);
	}

	auto back_propagate (const image<M, out_channels>& grad_Y) {
		image<N, in_channels> grad_X{};

		for (int f = 0; f < out_channels; f++)
			for (int i = 0; i < M; i++)
				for (int j = 0; j < M; j++)
					for (int x = 0; x < in_channels; x++)
						for (int y = 0; y < N; y++)
							for (int z = 0; z < N; z++)
								grad_X[x][y][z] += W[f][i][j][x][y][z] * grad_Y[f][i][j],
								W[f][i][j][x][y][z] += -eta * grad_Y[f][i][j] * last_X[x][y][z];

		return std::move(grad_X);
	}
};

struct model {
	ConvolutionalLayer<1, 32, 5, 1> conv1;
	ConvolutionalLayer<32, 32, 5, 1> conv2;
	ConvolutionalLayer<32, 64, 3, 1> conv3;
	ConvolutionalLayer<64, 64, 3, 1> conv4;
	FullyConnectedLayer<22, 64, 32, 2> fcon1;
	FullyConnectedLayer<32, 2, 1, 10> fcon2;

	MaxPool<2, 0> pool1;
	MaxPool<2, 0> pool2;
	ReLu relu;
	DropOut drop;

	auto evaluate (const image<28, 1>& img) {
		return
			fcon2.evaluate(
			relu.evaluate(
			fcon1.evaluate(
			pool2.evaluate(
		  	relu.evaluate(
		  	conv4.evaluate(
		  	conv3.evaluate(
			pool1.evaluate(
			relu.evaluate(
			conv2.evaluate(
			conv1.evaluate(img
				)))))))))));
	}

	auto back_propagate (const image<1, 10>& grad_Y) {
		conv1.back_propagate(
		conv2.back_propagate(
		relu.back_propagate(
		pool1.back_propagate(
		conv3.back_propagate(
		conv4.back_propagate(
		relu.back_propagate(
		pool2.back_propagate(
		fcon1.back_propagate(
		relu.back_propagate(
		fcon2.back_propagate(
			grad_Y)))))))))));
	}

	auto train (const pair<vector<image<28, 1>>, int>& T) {
		
	}
};


int main(){
    ios_base::sync_with_stdio(0), cin.tie(0);

    
    vector training_set(0, pair<image<28, 1>, int>());





}