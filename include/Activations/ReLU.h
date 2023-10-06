#ifndef NN_RELU_H_
#define NN_RELU_H_


namespace nn {

template<
	int kFeatures,
	int kChannels,
	class NextLayer
>
class ReLU {

public:

	ReLU ()
	{

	}


	auto recurse (nn::util::image<kFeatures, kChannels> X, const int label)
	{
		auto [gradient, loss] = L.recurse(forward(X), label);
		return std::pair(backward(X, gradient), loss);
	}


	auto evaluate (const nn::util::image<kFeatures, kChannels>& X, const int label) const
	{
		return L.evaluate(forward(X), label);
	}


	auto predict (const nn::util::image<kFeatures, kChannels>& X) const
	{
		return L.predict(forward(X));
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

	auto forward (nn::util::image<kFeatures, kChannels> X) const
	{

		for (int f = 0; f < kChannels; f++)
			for (int i = 0; i < kFeatures; i++)
				for (int j = 0; j < kFeatures; j++)
					if (X[f][i][j] < 0)
						X[f][i][j] = 0;

		return std::move(X);
	}


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