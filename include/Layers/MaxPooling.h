#ifndef NN_MAX_POOLING_H_
#define NN_MAX_POOLING_H_


namespace nn {

template<
	int kKernel,
	int kPadding,
	int kStride,
	class NextLayer
>
class MaxPool {
	static_assert(kPadding >= 0 and kKernel > kPadding and kStride > 0);

public:

	MaxPool ()
	{

	}


	template<uint64_t kInFeatures, uint64_t kChannels>
	auto recurse (const nn::util::image<kInFeatures, kChannels>& _X, const int label)
	{
		static constexpr int kInFeatures_ = kInFeatures + 2 * kPadding;
		static constexpr int kOutFeatures = (kInFeatures_ - kKernel + kStride) / kStride;

		auto X { nn::util::pad<kInFeatures, kChannels, kPadding>(_X) };

		auto [gradient, loss] = L.recurse(forward(X), label);
		return std::pair(backward(X, gradient), loss);
	}
	

	template<uint64_t kInFeatures, uint64_t kChannels>
	auto evaluate (const nn::util::image<kInFeatures, kChannels>& _X, const int label)
	{

		auto X { nn::util::pad<kInFeatures, kChannels, kPadding>(_X) };
		return L.evaluate(forward(X), label);
	}


	auto optimize ()
	{
		L.optimize();
	}


	auto save (const std::string path)
	{
		L.save(path);
	}


private:

	template<uint64_t kInFeatures_, uint64_t kChannels>
	auto forward (const nn::util::image<kInFeatures_, kChannels>& X) {
		static constexpr int kOutFeatures = (kInFeatures_ - kKernel + kStride) / kStride;

		nn::util::image<kOutFeatures, kChannels> Y { };

		for (int c = 0; c < kChannels; c++)
			for (int i = 0; i + kKernel <= kInFeatures_; i += kStride)
				for (int j = 0; j + kKernel <= kInFeatures_; j += kStride) {
					Y[c][i / kStride][j / kStride] = X[c][i][j];

					for (int x = 0; x < kKernel; x++)
						for (int y = 0; y < kKernel; y++)
							Y[c][i / kStride][j / kStride] = std::max(Y[c][i / kStride][j / kStride], X[c][i + x][j + y]);
				}

		return std::move(Y);
	}


	template<uint64_t kInFeatures_, uint64_t kOutFeatures, uint64_t kChannels>
	auto backward (const nn::util::image<kInFeatures_, kChannels>& X, const nn::util::image<kOutFeatures, kChannels>& grad_Y)
	{
		static_assert(kOutFeatures == (kInFeatures_ - kKernel + kStride) / kStride);

		nn::util::image<kInFeatures_, kChannels> grad_X { };

		for (int c = 0; c < kChannels; c++)
			for (int i = 0; i + kKernel <= kInFeatures_; i += kStride)
				for (int j = 0; j + kKernel <= kInFeatures_; j += kStride) {
					auto v = X[c][i][j];

					for (int x = 0; x < kKernel; x++)
						for (int y = 0; y < kKernel; y++)
							v = std::max(v, X[c][i + x][j + y]);
					for (int x = 0; x < kKernel; x++)
						for (int y = 0; y < kKernel; y++)
							if (v - X[c][i + x][j + y] < 1e-7)
								grad_X[c][i + x][j + y] += grad_Y[c][i / kStride][j / kStride];
				}

		return nn::util::pad<kInFeatures_, kChannels, -kPadding>(grad_X);
	}


	NextLayer L{ };

};

} // namespace nn

#endif // NN_MAX_POOLING_H_