struct ParametricReLU {
	// y = max(0, x);

	vector<vector<vector<double>>> last_X;
	double p{};

	ParametricReLU (const double a): p(a) {
		assert(0 < a and a < 1);
	}

	template<uint64_t N, uint64_t channels>
	auto evaluate (image<N, channels> X) {

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (X[f][i][j] < 0)
						X[f][i][j] *= p;

		return std::move(X);
	}

	template<uint64_t N, uint64_t channels>
	auto train (image<N, channels> X) {
		copy_to_vector(X, last_X);
		return evaluate(std::move(X));
	}

	template<uint64_t N, uint64_t channels>
	auto back_propagate (const image<N, channels>& grad_Y) {
		image<N, channels> grad_X{grad_Y};

		double grad_p{};

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (last_X[f][i][j] < 0)
						grad_X[f][i][j] *= p, grad_p += grad_Y[f][i][j] * last_X[f][i][j];

		p += -eps * grad_p;
		if (p < 0) p = 0;

		return std::move(grad_X);
	}
};