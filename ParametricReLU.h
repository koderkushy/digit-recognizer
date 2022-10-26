template<
	class Optimizer
>
struct ParametricReLU {
	vector<double> cache;
	double p{};

	ParametricReLU (const double a = 0.01) {
		assert(0 < a and a < 1);
		p = a;
	}

	template<uint64_t N, uint64_t channels>
	auto forward (image<N, channels> X) {

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (X[f][i][j] < 0)
						X[f][i][j] *= p;

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

		double grad_p{};

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (last_X[f][i][j] < 0)
						grad_X[f][i][j] *= p,
						grad_p += grad_Y[f][i][j] * last_X[f][i][j];

		static Optimizer optimizer{};
		optimizer.optimize(p, grad_p);

		if (p < 0) p = 0;

		return std::move(grad_X);
	}

	void save (const string path) {
		ofstream out(path);
		out << fixed << setprecision(10);
		out << p;
	}
};
