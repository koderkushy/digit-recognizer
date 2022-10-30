#ifndef NN_FULLY_CONNECTED_LAYER_H_
#define NN_FULLY_CONNECTED_LAYER_H_

// namespace nn {

template<
	class Optimizer,
	int N,
	int in_channels,
	int M,
	int out_channels,
	class NextLayer
>
struct FullyConnectedLayer {
	static constexpr int IN = N * N * in_channels, OUT = M * M * out_channels;

	array<array<double, OUT>, IN> W { };
	array<double, OUT> b { };
	NextLayer L { };

	decltype(W) grad_W_accumulate { };
	decltype(b) grad_b_accumulate { };
	int count_accumulate { };

	std::mutex grad_mutex;

	FullyConnectedLayer () {
		// Kaiming He initialisation

		mt19937 rng(chrono::high_resolution_clock::now().time_since_epoch().count());
		std::normal_distribution<double>
			gaussian { 0, sqrt(1.0 / IN) };

		for (int i = 0; i < IN; i++)
			for (int j = 0; j < OUT; j++)
				W[i][j] = gaussian(rng);

		for (int i = 0; i < OUT; i++)
			b[i] = gaussian(rng);
	}

	auto recurse (const image<N, in_channels>& X, const int label) {
		const auto [gradient, loss] = L.recurse(forward(X), label);
		return pair(backward(X, gradient), loss);
	}

	auto evaluate (const image<N, in_channels>& X, const int label) {
		return L.evaluate(forward(X), label);
	}

	auto forward (const image<N, in_channels>& X) {
			// auto start = chrono::high_resolution_clock::now();

		auto arr_X { array_converted(X) };
		array<double, OUT> arr_Y { b };

		for (int i = 0; i < IN; i++)
#pragma GCC ivdep
			for (int j = 0; j < OUT; j++)
				arr_Y[j] += W[i][j] * arr_X[i];

			// auto stop = chrono::high_resolution_clock::now();

			// cout << "fcon = " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << "ms\n" << flush;

		return imagify<M, out_channels, OUT>(arr_Y);
	}

	auto backward (const image<N, in_channels>& X, const image<M, out_channels>& grad_Y) {
		auto arr_grad_Y { array_converted(grad_Y) };
		auto arr_X { array_converted(X) };

		array<double, IN> arr_grad_X { };
		decltype(W) grad_W { };

		// Computing gradients wrt b
		const auto& grad_b { arr_grad_Y };

		// Computing gradients wrt X, W
		for (int i = 0; i < IN; i++)
#pragma GCC ivdep
			for (int j = 0; j < OUT; j++)
				arr_grad_X[i] += arr_grad_Y[j] * W[i][j],
				grad_W[i][j] = arr_grad_Y[j] * arr_X[i];

		{
			lock_guard<mutex> lock(grad_mutex);

			count_accumulate++;

#pragma GCC ivdep
			for (int i = 0; i < OUT; i++)
				grad_b_accumulate[i] += grad_b[i];

			for (int i = 0; i < IN; i++)
#pragma GCC ivdep
				for (int j = 0; j < OUT; j++)
					grad_W_accumulate[i][j] += grad_W[i][j];
		}

		return imagify<N, in_channels, IN>(arr_grad_X);
	}

	auto optimize () {
		static array<array<Optimizer, IN>, OUT> W_optimizer { };
		static array<Optimizer, OUT> b_optimizer { };

		{
			lock_guard<mutex> lock(grad_mutex);
			if (count_accumulate > 0) {
				for (int i = 0; i < OUT; i++)
					for (int j = 0; j < IN; j++)
						grad_W_accumulate[i][j] /= count_accumulate,
						W_optimizer[i][j].optimize(W[i][j], grad_W_accumulate[i][j]),
						grad_W_accumulate[i][j] = 0;

				for (int i = 0; i < OUT; i++)
					grad_b_accumulate[i] /= count_accumulate,
					b_optimizer[i].optimize(b[i], grad_b_accumulate[i]),
					grad_b_accumulate[i] = 0;

				count_accumulate = 0;

			}
		}
		L.optimize();
	}

	void save (const string path) {
		ofstream out(path, ios::app);

		out << "fcon " << OUT << ' ' << IN << '\n';

		for (int i = 0; i < OUT; i++)
			for (int j = 0; j < IN; j++)
				out << W[i][j] << ' ';
		out << '\n';
		for (int i = 0; i < OUT; i++)
			out << b[i] << ' ';
		out << '\n' << flush;

		L.save(path);
	}
};

// } // namespace nn

#endif // NN_FULLY_CONNECTED_LAYER_H_