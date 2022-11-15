#ifndef NN_DROP_OUT_H_
#define NN_DROP_OUT_H_


namespace nn {

template<
	int kPercent, class NextLayer
>
class DropOut {

	static constexpr uint64_t R = std::numeric_limits<uint32_t>::max();

	static_assert(0 <= kPercent and kPercent < 100);

public:

	DropOut (): rng(std::chrono::high_resolution_clock::now().time_since_epoch().count())
	{

	}


	template<uint64_t kFeatures, uint64_t kChannels>
	auto recurse (nn::util::image<kFeatures, kChannels> X, const int label)
	{
		std::array<bool, kFeatures * kFeatures * kChannels> mask { };

		for (auto& b: mask)
			b = (rng() > kPercent * R / 100.0);

		for (int f = 0; f < kChannels; f++)
			for (int i = 0; i < kFeatures; i++)
				for (int j = 0; j < kFeatures; j++)
					X[f][i][j] *= mask[(f * kFeatures + i) * kFeatures + j];

		auto [gradient, loss] = L.recurse(X, label);

		for (int f = 0; f < kChannels; f++)
			for (int i = 0; i < kFeatures; i++)
				for (int j = 0; j < kFeatures; j++)
					gradient[f][i][j] *= mask[(f * kFeatures + i) * kFeatures + j];

		return std::pair(std::move(gradient), loss);

	}


	template<uint64_t kFeatures, uint64_t kChannels>
	auto evaluate (const nn::util::image<kFeatures, kChannels>& X, const int label)
	{
		return L.evaluate(X, label);
	}


	template<uint64_t kFeatures, uint64_t kChannels>
	auto predict (const nn::util::image<kFeatures, kChannels>& X)
	{
		return L.predict(X);
	}


	auto save (const std::string path)
	{
		L.save(path);
	}


	auto optimize ()
	{
		L.optimize();
	}


private:

	std::mt19937 rng;

	NextLayer L{ };

};

} // namespace nn

#endif // NN_DROP_OUT_H_