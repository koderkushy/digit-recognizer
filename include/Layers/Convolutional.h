#ifndef NN_CONVOLUTIONAL_LAYER_H_
#define NN_CONVOLUTIONAL_LAYER_H_


namespace nn {

template<
	class Optimizer, int kInChannels, int kOutChannels, int kKernel, int kPadding, class NextLayer
>
class Convolutional {

public:

	Convolutional ()
	{
		// Kaiming He initialisation

		std::mt19937 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		std::normal_distribution<double> 
			gaussian { 0, sqrt(2.0 / (kKernel * kKernel * kInChannels)) };

		for (int o = 0; o < kOutChannels; o++)
			for (int i = 0; i < kInChannels; i++)
				for (int x = 0; x < kKernel; x++)
					for (int y = 0; y < kKernel; y++)
						W[o][i][x][y] = gaussian(rng);

		for (int f = 0; f < kOutChannels; f++)
			b[f] = gaussian(rng);

	}


	template<uint64_t kInFeatures>
	auto recurse (const nn::util::image<kInFeatures, kInChannels>& X, const int label)
	{
		const auto [gradient, loss] = L.recurse(forward(X), label);

		return std::pair(backward(X, gradient), loss);
	}


	template<uint64_t kInFeatures>
	auto evaluate (const nn::util::image<kInFeatures, kInChannels>& X, const int label)
	{
		return L.evaluate(forward(X), label);
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

	
	void save (const std::string path)
	{
		std::ofstream out(path, std::ios::app);

		out << "conv " << kOutChannels << ' ' << kInChannels << ' ' << kKernel << '\n';

		for (int o = 0; o < kOutChannels; o++)
			for (int i = 0; i < kInChannels; i++)
				for (int x = 0; x < kKernel; x++)
					for (int y = 0; y < kKernel; y++)
						out << W[o][i][x][y] << ' ';

		out << '\n';
		for (int o = 0; o < kOutChannels; o++)
			out << b[o] << ' ';
		out << '\n' << std::flush;

		L.save(path);
	}


private:

	template<uint64_t kInFeatures>
	auto forward (const nn::util::image<kInFeatures, kInChannels>& X_unpadded)
	{
		static constexpr int kOutFeatures = kInFeatures + (kPadding * 2) - kKernel + 1;
			// auto start = chrono::high_resolution_clock::now();

		auto Y { math::FastMath::convolve(nn::util::pad<kInFeatures, kInChannels, kPadding>(X_unpadded), W) };
		
		for (int f = 0; f < kOutChannels; f++)
			for (int i = 0; i < kOutFeatures; i++)
#pragma GCC ivdep
				for (int j = 0; j < kOutFeatures; j++)
					Y[f][i][j] += b[f];

			// auto stop = chrono::high_resolution_clock::now();
			// cout << "conv = " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << "ms\n" << flush;

		return std::move(Y);
	}


	template<uint64_t kInFeatures, uint64_t kOutFeatures>
	auto backward (const nn::util::image<kInFeatures, kInChannels>& X, const nn::util::image<kOutFeatures, kOutChannels>& grad_Y)
	{
		static_assert(kInFeatures == kOutFeatures - kPadding * 2 + kKernel - 1);

		auto W_flipped { W };

		decltype(W) grad_W { };
		decltype(b) grad_b { };

		nn::util::image<kInFeatures, kInChannels> grad_X { };

		// Compute gradients wrt W
		for (int g = 0; g < kInChannels; g++) {
			auto X_g { nn::util::pad<kInFeatures, kPadding>(X[g]) };

			for (int f = 0; f < kOutChannels; f++)
				grad_W[f][g] = math::FastMath::convolve(X_g, grad_Y[f])[0];
		}

		// Flip W
		for (int f = 0; f < kOutChannels; f++)
			for (int g = 0; g < kInChannels; g++)
				for (int i = 0; i < kKernel; i++) {
					for (int j = 0; j < kKernel; j++)
						std::reverse(W_flipped[f][g][i].begin(), W_flipped[f][g][i].end());
					std::reverse(W_flipped[f][g].begin(), W_flipped[f][g].end());
				}

		// Compute gradients wrt X
		auto grad_Y_padded { nn::util::pad<kOutFeatures, kOutChannels, kKernel - 1>(grad_Y) };

		for (int g = 0; g < kInChannels; g++) {
			for (int f = 0; f < kOutChannels; f++) {
				auto component {math::FastMath::convolve(grad_Y_padded[f], W_flipped[f][g])[0]};

				for (int i = 0; i < kInFeatures; i++)
#pragma GCC ivdep
					for (int j = 0; j < kInFeatures; j++)
						grad_X[g][i][j] += component[i + kPadding][j + kPadding];
			}
		}

		// Compute gradients wrt b
		for (int f = 0; f < kOutChannels; f++)
			for (int i = 0; i < kOutFeatures; i++)
#pragma GCC ivdep
				for (int j = 0; j < kOutFeatures; j++)
					grad_b[f] += grad_Y[f][i][j];

		// Acquire mutex and accumulate gradients
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

		return std::move(grad_X);
	}


	std::array<nn::util::image<kKernel, kInChannels>, kOutChannels> W{ };

	std::array<double, kOutChannels> b{ };

	NextLayer L{ };

	std::array<nn::util::image<kKernel, kInChannels>, kOutChannels> grad_W_accumulate{ };
	
	std::array<double, kOutChannels> grad_b_accumulate{ };

	int count_accumulate{ };

	std::mutex grad_mutex{ };

};

} // namespace nn

#endif // NN_CONVOLUTIONAL_LAYER_H_
