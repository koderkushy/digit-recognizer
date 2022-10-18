template<int N, int in_channels, int M, int out_channels>
struct FullyConnectedLayer {

	array<array<array<image<N, in_channels>, M>, M>, out_channels> W;
	vector<vector<vector<double>>> last_X;

	FullyConnectedLayer () {
		// Kaiming He initialisation
		mt19937 rng(chrono::high_resolution_clock::now().time_since_epoch().count());
		constexpr double stddev = sqrt(2.0 / (N * N * in_channels))
		std::normal_distribution<double> N{0, stddev};

		for (int o = 0; o < out_channels; o++)
			for (int p = 0; p < M; p++)
				for (int q = 0; q < M; q++)
					for (int i = 0; i < in_channels; i++)
						for (int x = 0; x < N; x++)
							for (int y = 0; y < N; y++)
								W[o][p][q][i][x][y] = N(rng);
	}

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
