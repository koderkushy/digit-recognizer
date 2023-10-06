#ifndef NN_DROP_OUT_H_
#define NN_DROP_OUT_H_


namespace nn {

template<
	int kSize,
	int kChannels,
	int kPercent,
	class NextLayer
>
class DropOut {

	static constexpr uint64_t R = std::numeric_limits<uint32_t>::max();

	static_assert(0 <= kPercent and kPercent < 100);

public:

	DropOut (): rng(std::chrono::high_resolution_clock::now().time_since_epoch().count())
	{

	}


	auto recurse (nn::util::image<kSize, kChannels> X, const int label)
	{
		std::array<bool, kSize * kSize * kChannels> mask { };

		for (auto& b: mask)
			b = (rng() > kPercent * R / 100.0);

		for (int f = 0; f < kChannels; f++)
			for (int i = 0; i < kSize; i++)
				for (int j = 0; j < kSize; j++)
					X[f][i][j] *= mask[(f * kSize + i) * kSize + j];

		auto [gradient, loss] = L.recurse(X, label);

		for (int f = 0; f < kChannels; f++)
			for (int i = 0; i < kSize; i++)
				for (int j = 0; j < kSize; j++)
					gradient[f][i][j] *= mask[(f * kSize + i) * kSize + j];

		return std::pair(std::move(gradient), loss);

	}


	auto evaluate (const nn::util::image<kSize, kChannels>& X, const int label) const
	{
		return L.evaluate(X, label);
	}


	auto predict (const nn::util::image<kSize, kChannels>& X) const
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