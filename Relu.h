struct ReLU {
	// y = max(0, x);

	vector<vector<vector<double>>> last_X;

	template<uint64_t N, uint64_t channels>
	auto forward (image<N, channels> X) {

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
		return forward(std::move(X));
	}

	template<uint64_t N, uint64_t channels>
	auto backward (const image<N, channels>& grad_Y) {
		image<N, channels> grad_X{};

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (last_X[f][i][j] > 0)
						grad_X[f][i][j] = grad_Y[f][i][j];

		return std::move(grad_X);
	}
};