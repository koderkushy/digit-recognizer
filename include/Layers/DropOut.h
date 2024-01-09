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


	void save (const std::string path, const int layer_index = 0) const
	{
		std::ofstream desc_out(path + "model_description.txt"
									, std::ios::out | std::ios::app);
		std::ofstream lyr_stream(path + "layer-" + std::to_string(layer_index) + ".bin"
									, std::ios::out | std::ios::binary);

		desc_out << "drop out\n"
			<< "size " << kSize << '\n'
			<< "channels " << kChannels << '\n'
			<< "drop out percent " << kPercent << '\n' << std::flush;
		
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


	auto optimize ()
	{
		L.optimize();
	}

	auto size () const {
		return L.size();
	}


private:

	std::mt19937 rng;

	NextLayer L{ };

};

} // namespace nn

#endif // NN_DROP_OUT_H_