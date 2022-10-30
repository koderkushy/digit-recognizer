#ifndef NN_CONVOLUTIONAL_LAYER_H_
#define NN_CONVOLUTIONAL_LAYER_H_

#include "../Operations/trivial.h"
#include "../Operations/matmul.h"

namespace nn {

template<
	class Optimizer,
	int in_channels,
	int out_channels,
	int kernel_width,
	int padding_width,
	class NextLayer
>
struct ConvolutionalLayer {

	array<nn::operations::image<kernel_width, in_channels>, out_channels> W { };
	array<double, out_channels> b { };
	NextLayer L {};

	decltype(W) grad_W_accumulate { };
	decltype(b) grad_b_accumulate { };
	int count_accumulate { };

	std::mutex grad_mutex;

	ConvolutionalLayer () {
		// Kaiming He initialisation

		mt19937 rng(chrono::high_resolution_clock::now().time_since_epoch().count());
		std::normal_distribution<double> 
			gaussian { 0, sqrt(2.0 / (kernel_width * kernel_width * in_channels)) };

		for (int o = 0; o < out_channels; o++)
			for (int i = 0; i < in_channels; i++)
				for (int x = 0; x < kernel_width; x++)
					for (int y = 0; y < kernel_width; y++)
						W[o][i][x][y] = gaussian(rng);

		for (int f = 0; f < out_channels; f++)
			b[f] = gaussian(rng);

	}

	template<uint64_t N>
	auto recurse (const nn::operations::image<N, in_channels>& X, const int label) {
		const auto [gradient, loss] = L.recurse(forward(X), label);

		return pair(backward(X, gradient), loss);
	}

	template<uint64_t N>
	auto evaluate (const nn::operations::image<N, in_channels>& X, const int label) {
		return L.evaluate(forward(X), label);
	}

	template<uint64_t N>
	auto forward (const nn::operations::image<N, in_channels>& X_unpadded) {
		static constexpr int M = N + (padding_width * 2) - kernel_width + 1;
			auto start = chrono::high_resolution_clock::now();

		auto Y { operations::Math::convolve(nn::operations::pad<N, in_channels, padding_width>(X_unpadded), W) };
		
		for (int f = 0; f < out_channels; f++)
			for (int i = 0; i < M; i++)
#pragma GCC ivdep
				for (int j = 0; j < M; j++)
					Y[f][i][j] += b[f];

			auto stop = chrono::high_resolution_clock::now();
			cout << "conv = " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << "ms\n" << flush;

		return std::move(Y);
	}

	template<uint64_t N, uint64_t M>
	auto backward (const nn::operations::image<N, in_channels>& X, const nn::operations::image<M, out_channels>& grad_Y) {
		static_assert(N == M - padding_width * 2 + kernel_width - 1);

		auto W_flipped { W };

		decltype(W) grad_W { };
		decltype(b) grad_b { };
		nn::operations::image<N, in_channels> grad_X { };

		// Compute gradients wrt W
		for (int g = 0; g < in_channels; g++) {
			auto X_g { nn::operations::pad<N, padding_width>(X[g]) };

			for (int f = 0; f < out_channels; f++)
				grad_W[f][g] = nn::operations::Math::convolve(X_g, grad_Y[f])[0];
		}

		// Flip W
		for (int f = 0; f < out_channels; f++)
			for (int g = 0; g < in_channels; g++)
				for (int i = 0; i < kernel_width; i++) {
					for (int j = 0; j < kernel_width; j++)
						reverse(W_flipped[f][g][i].begin(), W_flipped[f][g][i].end());
					reverse(W_flipped[f][g].begin(), W_flipped[f][g].end());
				}

		// Compute gradients wrt X
		auto grad_Y_padded { nn::operations::pad<M, out_channels, kernel_width - 1>(grad_Y) };

		for (int g = 0; g < in_channels; g++) {
			for (int f = 0; f < out_channels; f++) {
				auto component {nn::operations::Math::convolve(grad_Y_padded[f], W_flipped[f][g])[0]};

				for (int i = 0; i < N; i++)
#pragma GCC ivdep
					for (int j = 0; j < N; j++)
						grad_X[g][i][j] += component[i + padding_width][j + padding_width];
			}
		}

		// Compute gradients wrt b
		for (int f = 0; f < out_channels; f++)
			for (int i = 0; i < M; i++)
#pragma GCC ivdep
				for (int j = 0; j < M; j++)
					grad_b[f] += grad_Y[f][i][j];

		// Acquire mutex and accumulate gradients
		{
			lock_guard<mutex> lock(grad_mutex);

			count_accumulate++;
			for (int o = 0; o < out_channels; o++)
				for (int i = 0; i < in_channels; i++)
					for (int x = 0; x < kernel_width; x++)
#pragma GCC ivdep
						for (int y = 0; y < kernel_width; y++)
							grad_W_accumulate[o][i][x][y] += grad_W[o][i][x][y];

#pragma GCC ivdep
			for (int f = 0; f < out_channels; f++)
				grad_b_accumulate[f] += grad_b[f];
		}

		return std::move(grad_X);
	}

	auto optimize () {
		static array<array<array<array<Optimizer, kernel_width>, kernel_width>, in_channels>, out_channels> W_optimizer { };
		static array<Optimizer, out_channels> b_optimizer { };

		
		{
			lock_guard<mutex> lock(grad_mutex);

			if (count_accumulate > 0) {
				for (int o = 0; o < out_channels; o++)
					for (int i = 0; i < in_channels; i++)
						for (int x = 0; x < kernel_width; x++)
							for (int y = 0; y < kernel_width; y++)
								grad_W_accumulate[o][i][x][y] /= count_accumulate,
								W_optimizer[o][i][x][y].optimize(W[o][i][x][y], grad_W_accumulate[o][i][x][y]),
								grad_W_accumulate[o][i][x][y] = 0;

				for (int f = 0; f < out_channels; f++)
					grad_b_accumulate[f] /= count_accumulate,
					b_optimizer[f].optimize(b[f], grad_b_accumulate[f]),
					grad_b_accumulate[f] = 0;

				count_accumulate = 0;
			}
		}

		L.optimize();
	}
	
	void save (const string path) {
		ofstream out(path, ios::app);

		out << "conv " << out_channels << ' ' << in_channels << ' ' << kernel_width << '\n';

		for (int o = 0; o < out_channels; o++)
			for (int i = 0; i < in_channels; i++)
				for (int x = 0; x < kernel_width; x++)
					for (int y = 0; y < kernel_width; y++)
						out << W[o][i][x][y] << ' ';

		out << '\n';
		for (int o = 0; o < out_channels; o++)
			out << b[o] << ' ';
		out << '\n' << flush;

		L.save(path);
	}
};

} // namespace nn

#endif // NN_CONVOLUTIONAL_LAYER_H_
