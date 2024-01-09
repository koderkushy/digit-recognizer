#ifndef NN_CONVOLUTIONAL_LAYER_H_
#define NN_CONVOLUTIONAL_LAYER_H_

using uint = long unsigned int;

namespace nn {

template<
	int kInFeatures,
	int kInChannels,
	int kKernel,
	int kPadding,
	int kOutChannels,
	class Optimizer,
	class NextLayer
>
class Convolutional {
	static constexpr int kPaddedSize = kInFeatures + 2 * kPadding;
	static constexpr int kOutFeatures = kPaddedSize - kKernel + 1;
public:

	Convolutional ()
	{
		// Kaiming He initialisation

		std::mt19937 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		std::normal_distribution<float> 
			gaussian { 0, sqrt(2.0 / (kKernel * kKernel * kInChannels)) };

		for (int o = 0; o < kOutChannels; o++)
			for (int i = 0; i < kInChannels; i++)
				for (int x = 0; x < kKernel; x++)
					for (int y = 0; y < kKernel; y++)
						W[o][i][x][y] = gaussian(rng);

		for (int f = 0; f < kOutChannels; f++)
			b[f] = gaussian(rng);

	}


	auto recurse (const nn::util::image<kInFeatures, kInChannels>& X, const int label)
	{
		const auto [gradient, loss] = L.recurse(forward(X), label);

		return std::pair(backward(X, gradient), loss);
	}


	auto evaluate (const nn::util::image<kInFeatures, kInChannels>& X, const int label) const
	{
		return L.evaluate(forward(X), label);
	}


	auto predict (const nn::util::image<kInFeatures, kInChannels>& X) const
	{
		return L.predict(forward(X));
	}


	auto optimize ()
	{
		static std::array<std::array<std::array<std::array<Optimizer, kKernel>, kKernel>, kInChannels>, kOutChannels> W_optimizer { };
		static std::array<Optimizer, kOutChannels> b_optimizer { };

		
		{
			std::lock_guard<std::mutex> lock(grad_mutex);

			if (count_accumulate > 0) {
				for (int o = 0; o < kOutChannels; o++)
					for (int i = 0; i < kInChannels; i++)
						for (int x = 0; x < kKernel; x++)
							for (int y = 0; y < kKernel; y++)
								grad_W_accumulate[o][i][x][y] /= count_accumulate,
								W_optimizer[o][i][x][y].optimize(W[o][i][x][y], grad_W_accumulate[o][i][x][y]),
								grad_W_accumulate[o][i][x][y] = 0;

				for (int f = 0; f < kOutChannels; f++)
					grad_b_accumulate[f] /= count_accumulate,
					b_optimizer[f].optimize(b[f], grad_b_accumulate[f]),
					grad_b_accumulate[f] = 0;

				count_accumulate = 0;
			}
		}

		L.optimize();
	}

	
	void save (const std::string path, const int layer_index = 0) const
	{
		std::ofstream desc_out(path + "model_description.txt"
									, std::ios::out | std::ios::app);
		std::ofstream lyr_stream(path + "layer-" + std::to_string(layer_index) + ".bin"
									, std::ios::out | std::ios::binary);

		desc_out << "conv\n"
			<< "input features " << kInFeatures << '\n'
			<< "input channels " << kInChannels << '\n'
			<< "kernel size " << kKernel << '\n'
			<< "padding " << kPadding << '\n'
			<< "output channels " << kOutChannels << '\n' << std::flush;
		desc_out.close();
    	
    	lyr_stream.write(reinterpret_cast<const char*>(&W), sizeof(decltype(W)));
    	lyr_stream.write(reinterpret_cast<const char*>(&b), sizeof(decltype(b)));
		lyr_stream.close();

		L.save(path, layer_index + 1);
	}

	void load (const std::string path, const int layer_index = 0)
	{
		std::ifstream lyr_stream(path + "layer-" + std::to_string(layer_index) + ".bin"
									, std::ios::in | std::ios::binary);
		
		assert(lyr_stream.is_open());
		lyr_stream.read(reinterpret_cast<char*>(&W), sizeof(decltype(W)));
		lyr_stream.read(reinterpret_cast<char*>(&b), sizeof(decltype(b)));

		lyr_stream.close();

		L.load(path, layer_index + 1);
	}

	auto size () const {
		return (kKernel * kInChannels + 1) * kOutChannels * 2 + L.size();
	}


private:

	template<uint N, uint NC, uint K, uint KC>
	static auto convolve (const nn::util::image<N, NC>& X, const std::array<nn::util::image<K, NC>, KC>& W)
	{
		static constexpr int M = N - K + 1;
		nn::util::image<M, KC> Y{};

		std::array<std::array<float, K * K * NC>, KC> W_mat{};
		for (int f = 0; f < KC; f++)
			for (int g = 0; g < NC; g++)
				for (int i = 0; i < K; i++)
					for (int j = 0; j < K; j++)
						W_mat[f][(g * K + i) * K + j] = W[f][g][i][j];

		std::array<std::array<float, M * M>, K * K * NC> X_mat{};	
		for (int i = 0; i < M; i++)
			for (int j = 0; j < M; j++)
				for (int g = 0; g < NC; g++)
					for (int x = 0; x < K; x++)
						for (int y = 0; y < K; y++)
							X_mat[y + K * (x + K * g)][i * M + j] = X[g][i + x][j + y];

		const auto Y_mat {nn::math::FastMath::mat_mul<K * K * NC, KC, M * M>(W_mat, X_mat)};

		for (int f = 0; f < KC; f++)
			Y[f] = nn::util::imagify<M, 1, M * M>(Y_mat[f])[0];

		return std::move(Y);
	}

	template<uint N, uint K>
	static auto convolve (const nn::util::filter<N>& X, const nn::util::filter<K>& W)
	{
		nn::util::image<N, 1> _X{};
		std::array<nn::util::image<K, 1>, 1> _W{};
		_X[0] = X, _W[0][0] = W;
		return convolve(_X, _W);
	}

	auto forward (const nn::util::image<kInFeatures, kInChannels>& X_unpadded) const
	{
		static_assert(kInFeatures >= kKernel);
			// auto start = std::chrono::high_resolution_clock::now();

		auto X = nn::util::pad<kInFeatures, kInChannels, kPadding>(X_unpadded);
		auto Y { convolve<kPaddedSize, kInChannels, kKernel, kOutChannels>(X, W) };
		
		for (int f = 0; f < kOutChannels; f++)
			for (int i = 0; i < kOutFeatures; i++)
#pragma GCC ivdep
				for (int j = 0; j < kOutFeatures; j++)
					Y[f][i][j] += b[f];

			// auto stop = std::chrono::high_resolution_clock::now();

			// std::cout << "conv fwd = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms\n" << std::flush;

		return std::move(Y);
	}


	auto backward (const nn::util::image<kInFeatures, kInChannels>& X, const nn::util::image<kOutFeatures, kOutChannels>& grad_Y)
	{
			// auto start = std::chrono::high_resolution_clock::now();

		auto W_flipped { W };

		std::array<nn::util::image<kKernel, kInChannels>, kOutChannels> grad_W { };
		std::array<float, kOutChannels> grad_b { };

		nn::util::image<kInFeatures, kInChannels> grad_X { };

		{
			int g = 0;
			for (; g + 4 <= kInChannels; g += 4) {
				const auto X0 { nn::util::pad<kInFeatures, kPadding>(X[g + 0]) };
				const auto X1 { nn::util::pad<kInFeatures, kPadding>(X[g + 1]) };
				const auto X2 { nn::util::pad<kInFeatures, kPadding>(X[g + 2]) };
				const auto X3 { nn::util::pad<kInFeatures, kPadding>(X[g + 3]) };
#pragma GCC ivdep
				for (int f = 0; f < kOutChannels; f++) {
					grad_W[f][g + 0] = convolve(X0, grad_Y[f])[0];
					grad_W[f][g + 1] = convolve(X1, grad_Y[f])[0];
					grad_W[f][g + 2] = convolve(X2, grad_Y[f])[0];
					grad_W[f][g + 3] = convolve(X3, grad_Y[f])[0];
				}
			}

			for (; g < kInChannels; g++) {
				const auto X_g { nn::util::pad<kInFeatures, kPadding>(X[g]) };
#pragma GCC ivdep
				for (int f = 0; f < kOutChannels; f++)
					grad_W[f][g] = convolve(X_g, grad_Y[f])[0];
			}
			
		}

		for (int f = 0; f < kOutChannels; f++)
			for (int g = 0; g < kInChannels; g++)
				for (int i = 0; i < kKernel; i++) {
					for (int j = 0; j < kKernel; j++)
						std::reverse(W_flipped[f][g][i].begin(), W_flipped[f][g][i].end());
					std::reverse(W_flipped[f][g].begin(), W_flipped[f][g].end());
				}

		auto grad_Y_padded { nn::util::pad<kOutFeatures, kOutChannels, kKernel - 1>(grad_Y) };

		for (int g = 0; g < kInChannels; g++) {
			for (int f = 0; f < kOutChannels; f++) {
				const auto component {convolve(grad_Y_padded[f], W_flipped[f][g])[0]};

				for (int i = 0; i < kInFeatures; i++)
#pragma GCC ivdep
					for (int j = 0; j < kInFeatures; j++)
						grad_X[g][i][j] += component[i + kPadding][j + kPadding];
			}
		}

		for (int f = 0; f < kOutChannels; f++)
			for (int i = 0; i < kOutFeatures; i++)
#pragma GCC ivdep
				for (int j = 0; j < kOutFeatures; j++)
					grad_b[f] += grad_Y[f][i][j];

		{
			std::lock_guard<std::mutex> lock(grad_mutex);

			count_accumulate++;
			for (int o = 0; o < kOutChannels; o++)
				for (int i = 0; i < kInChannels; i++)
					for (int x = 0; x < kKernel; x++)
#pragma GCC ivdep
						for (int y = 0; y < kKernel; y++)
							grad_W_accumulate[o][i][x][y] += grad_W[o][i][x][y];

#pragma GCC ivdep
			for (int f = 0; f < kOutChannels; f++)
				grad_b_accumulate[f] += grad_b[f];
		}

			// auto stop = std::chrono::high_resolution_clock::now();

			// std::cout << "conv bwd = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms\n" << std::flush;

		return std::move(grad_X);
	}


	std::array<nn::util::image<kKernel, kInChannels>, kOutChannels> W{ };

	std::array<float, kOutChannels> b{ };

	NextLayer L{ };

	std::array<nn::util::image<kKernel, kInChannels>, kOutChannels> grad_W_accumulate{ };
	
	std::array<float, kOutChannels> grad_b_accumulate{ };

	int count_accumulate{ };

	std::mutex grad_mutex{ };

};

} // namespace nn

#endif // NN_CONVOLUTIONAL_LAYER_H_
