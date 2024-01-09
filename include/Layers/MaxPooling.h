#ifndef NN_MAX_POOLING_H_
#define NN_MAX_POOLING_H_


namespace nn {

template<
	int kInFeatures,
	int kChannels,
	int kKernel,
	int kPadding,
	int kStride,
	class NextLayer
>
class MaxPool {
	static_assert(kPadding >= 0 and kKernel > kPadding and kStride > 0);

	static constexpr int kInFeaturesPadded = kInFeatures + 2 * kPadding;
	static constexpr int kOutFeatures = (kInFeaturesPadded - kKernel + kStride) / kStride;

public:

	MaxPool ()
	{

	}


	auto recurse (const nn::util::image<kInFeatures, kChannels>& _X, const int label)
	{
		auto X { nn::util::pad<kInFeatures, kChannels, kPadding>(_X) };

		auto [gradient, loss] = L.recurse(forward(X), label);
		return std::pair(backward(X, gradient), loss);
	}
	

	auto evaluate (const nn::util::image<kInFeatures, kChannels>& _X, const int label) const
	{
		auto X { nn::util::pad<kInFeatures, kChannels, kPadding>(_X) };
		return L.evaluate(forward(X), label);
	}


	auto predict (const nn::util::image<kInFeatures, kChannels>& _X) const
	{
		auto X { nn::util::pad<kInFeatures, kChannels, kPadding>(_X) };
		return L.predict(forward(X));
	}


	auto optimize ()
	{
		L.optimize();
	}


	void save (const std::string path, const int layer_index = 0) const
	{
		std::ofstream desc_out(path + "model_description.txt"
									, std::ios::out | std::ios::app);
		std::ofstream lyr_stream(path + "layer-" + std::to_string(layer_index) + ".bin"
									, std::ios::out | std::ios::binary);

		desc_out << "max pooling\n"
			<< "input features " << kInFeatures << '\n'
			<< "channels " << kChannels << '\n'
			<< "kernel size " << kKernel << '\n'
			<< "padding " << kPadding << '\n'
			<< "stride " << kStride << '\n' << std::flush;
		
		desc_out.close();
		lyr_stream.close();

		L.save(path, layer_index + 1);
	}

	void load (const std::string path, const int layer_index = 0)
	{
		std::ifstream lyr_stream(path + "layer-" + std::to_string(layer_index) + ".bin"
									, std::ios::in | std::ios::binary);
		
		assert(lyr_stream.is_open());
		lyr_stream.close();

		L.load(path, layer_index + 1);
	}

	auto size () const {
		return L.size();
	}


private:

	auto forward (const nn::util::image<kInFeaturesPadded, kChannels>& X) const
	{
		static constexpr int kOutFeatures = (kInFeaturesPadded - kKernel) / kStride + 1;

		nn::util::image<kOutFeatures, kChannels> Y { };

		for (int c = 0; c < kChannels; c++)
			for (int i = 0; i + kKernel <= kInFeaturesPadded; i += kStride)
				for (int j = 0; j + kKernel <= kInFeaturesPadded; j += kStride) {
					Y[c][i / kStride][j / kStride] = X[c][i][j];

					for (int x = 0; x < kKernel; x++)
						for (int y = 0; y < kKernel; y++)
							Y[c][i / kStride][j / kStride] = std::max(Y[c][i / kStride][j / kStride], X[c][i + x][j + y]);
				}

		return std::move(Y);
	}


	auto backward (const nn::util::image<kInFeaturesPadded, kChannels>& X, const nn::util::image<kOutFeatures, kChannels>& grad_Y)
	{
		static_assert(kOutFeatures == (kInFeaturesPadded - kKernel) / kStride + 1);

		nn::util::image<kInFeaturesPadded, kChannels> grad_X { };

		for (int c = 0; c < kChannels; c++)
			for (int i = 0; i + kKernel <= kInFeaturesPadded; i += kStride)
				for (int j = 0; j + kKernel <= kInFeaturesPadded; j += kStride) {
					auto v = X[c][i][j];

					for (int x = 0; x < kKernel; x++)
						for (int y = 0; y < kKernel; y++)
							v = std::max(v, X[c][i + x][j + y]);
					for (int x = 0; x < kKernel; x++)
						for (int y = 0; y < kKernel; y++)
							if (v - X[c][i + x][j + y] < 1e-7)
								grad_X[c][i + x][j + y] += grad_Y[c][i / kStride][j / kStride];
				}

		return nn::util::pad<kInFeaturesPadded, kChannels, -kPadding>(grad_X);
	}


	NextLayer L{ };

};

} // namespace nn

#endif // NN_MAX_POOLING_H_