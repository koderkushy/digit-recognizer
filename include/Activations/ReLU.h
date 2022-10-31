#ifndef NN_RELU_H_
#define NN_RELU_H_


namespace nn {

template<
	class NextLayer
>
class ReLU {

public:

	ReLU ()
	{

	}


	template<uint64_t kFeatures, uint64_t kChannels>
	auto recurse (nn::util::image<kFeatures, kChannels> X, const int label)
	{
		auto [gradient, loss] = L.recurse(forward(X), label);
		return std::pair(backward(X, gradient), loss);
	}


	template<uint64_t kFeatures, uint64_t kChannels>
	auto evaluate (const nn::util::image<kFeatures, kChannels>& X, const int label)
	{
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

	template<uint64_t kFeatures, uint64_t kChannels>
	auto forward (nn::util::image<kFeatures, kChannels> X)
	{

		for (int f = 0; f < kChannels; f++)
			for (int i = 0; i < kFeatures; i++)
				for (int j = 0; j < kFeatures; j++)
					if (X[f][i][j] < 0)
						X[f][i][j] = 0;

		return std::move(X);
	}


	template<uint64_t kFeatures, uint64_t kChannels>
	auto backward (const nn::util::image<kFeatures, kChannels>& X, const nn::util::image<kFeatures, kChannels>& grad_Y)
	{
		nn::util::image<kFeatures, kChannels> grad_X{};

		for (int f = 0; f < kChannels; f++)
			for (int i = 0; i < kFeatures; i++)
				for (int j = 0; j < kFeatures; j++)
					if (X[f][i][j] > 0)
						grad_X[f][i][j] = grad_Y[f][i][j];

		return std::move(grad_X);
	}


	NextLayer L{ };

};

} // namespace nn

#endif // NN_RELU_H_