template<
	template<int n, int c> class Optimizer
>
struct ParametricReLU {
	// y = max(0, x);

	vector<double> cache;
	image<1, 1> p{};

	ParametricReLU (const double a) {
		assert(0 < a and a < 1);
		p[0][0][0] = a;
	}

	template<uint64_t N, uint64_t channels>
	auto forward (image<N, channels> X) {

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (X[f][i][j] < 0)
						X[f][i][j] *= p[0][0][0];

		return std::move(X);
	}

	template<uint64_t N, uint64_t channels>
	auto train (image<N, channels> X) {
		copy_to_vector(X, cache);
		return forward(std::move(X));
	}

	template<uint64_t N, uint64_t channels>
	auto backward (const image<N, channels>& grad_Y) {
		auto grad_X{grad_Y};
		auto last_X{imagify<N, channels>(cache)};

		image<1, 1> grad_p{};

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (last_X[f][i][j] < 0)
						grad_X[f][i][j] *= p[0][0][0],
						grad_p[0][0][0] += grad_Y[f][i][j] * last_X[f][i][j];

		static Optimizer<1, 1> optimizer{};
		optimizer.optimize(p, grad_p);

		if (p[0][0][0] < 0) p[0][0][0] = 0;

		return std::move(grad_X);
	}
};
