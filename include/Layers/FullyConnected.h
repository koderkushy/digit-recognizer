#ifndef NN_FULLY_CONNECTED_LAYER_H_
#define NN_FULLY_CONNECTED_LAYER_H_


namespace nn {

template<
	class Optimizer, int kInWidth, int kOutWidth, class NextLayer
>
class FullyConnected {

public:

	FullyConnected ()
	{
		// Kaiming He initialisation

		std::mt19937 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		std::normal_distribution<double>
			gaussian { 0, sqrt(1.0 / kInWidth) };

		for (int i = 0; i < kInWidth; i++)
			for (int j = 0; j < kOutWidth; j++)
				W[i][j] = gaussian(rng);

		for (int i = 0; i < kOutWidth; i++)
			b[i] = gaussian(rng);
	}


	template<uint64_t kFeatures, uint64_t kChannels>
	auto recurse (const nn::util::image<kFeatures, kChannels>& X, const int label)
	{
		const auto [gradient, loss] = L.recurse(forward(X), label);
		return std::pair(backward(X, gradient), loss);
	}


	template<uint64_t kFeatures, uint64_t kChannels>
	auto evaluate (const nn::util::image<kFeatures, kChannels>& X, const int label)
	{
		return L.evaluate(forward(X), label);
	}


	auto optimize ()
	{
		static std::array<std::array<Optimizer, kInWidth>, kOutWidth> W_optimizer { };
		static std::array<Optimizer, kOutWidth> b_optimizer { };

		{
			std::lock_guard<std::mutex> lock(grad_mutex);
			if (count_accumulate > 0) {
				for (int i = 0; i < kOutWidth; i++)
					for (int j = 0; j < kInWidth; j++)
						grad_W_accumulate[i][j] /= count_accumulate,
						W_optimizer[i][j].optimize(W[i][j], grad_W_accumulate[i][j]),
						grad_W_accumulate[i][j] = 0;

				for (int i = 0; i < kOutWidth; i++)
					grad_b_accumulate[i] /= count_accumulate,
					b_optimizer[i].optimize(b[i], grad_b_accumulate[i]),
					grad_b_accumulate[i] = 0;

				count_accumulate = 0;

			}
		}
		L.optimize();
	}


	void save (const std::string path)
	{
		std::ofstream out(path, std::ios::app);

		out << "fcon " << kOutWidth << ' ' << kInWidth << '\n';

		for (int i = 0; i < kOutWidth; i++)
			for (int j = 0; j < kInWidth; j++)
				out << W[i][j] << ' ';
		out << '\n';
		for (int i = 0; i < kOutWidth; i++)
			out << b[i] << ' ';
		out << '\n' << std::flush;

		L.save(path);
	}


private:

	template<uint64_t kFeatures, uint64_t kChannels>
	auto forward (const nn::util::image<kFeatures, kChannels>& X)
	{

		static_assert(kFeatures * kFeatures * kChannels == kInWidth);
			// auto start = chrono::high_resolution_clock::now();

		auto arr_X { nn::util::array_converted(X) };
		std::array<double, kOutWidth> arr_Y { b };

		for (int i = 0; i < kInWidth; i++)
#pragma GCC ivdep
			for (int j = 0; j < kOutWidth; j++)
				arr_Y[j] += W[i][j] * arr_X[i];

			// auto stop = chrono::high_resolution_clock::now();
			// cout << "fcon = " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << "ms\n" << flush;

		return nn::util::imagify<1, kOutWidth, kOutWidth>(arr_Y);
	}


	template<uint64_t kFeatures, uint64_t kChannels>
	auto backward (const nn::util::image<kFeatures, kChannels>& X, const nn::util::image<1, kOutWidth>& grad_Y)
	{
		auto arr_grad_Y { nn::util::array_converted(grad_Y) };
		auto arr_X { nn::util::array_converted(X) };

		std::array<double, kInWidth> arr_grad_X { };
		decltype(W) grad_W { };

		// Computing gradients wrt b
		const auto& grad_b { arr_grad_Y };

		// Computing gradients wrt X, W
		for (int i = 0; i < kInWidth; i++)
#pragma GCC ivdep
			for (int j = 0; j < kOutWidth; j++)
				arr_grad_X[i] += arr_grad_Y[j] * W[i][j],
				grad_W[i][j] = arr_grad_Y[j] * arr_X[i];

		{
			std::lock_guard<std::mutex> lock(grad_mutex);

			count_accumulate++;

#pragma GCC ivdep
			for (int i = 0; i < kOutWidth; i++)
				grad_b_accumulate[i] += grad_b[i];

			for (int i = 0; i < kInWidth; i++)
#pragma GCC ivdep
				for (int j = 0; j < kOutWidth; j++)
					grad_W_accumulate[i][j] += grad_W[i][j];
		}

		return nn::util::imagify<kFeatures, kChannels, kInWidth>(arr_grad_X);
	}


	std::array<std::array<double, kOutWidth>, kInWidth> W{ };

	std::array<double, kOutWidth> b{ };

	NextLayer L{ };

	std::array<std::array<double, kOutWidth>, kInWidth> grad_W_accumulate{ };

	std::array<double, kOutWidth> grad_b_accumulate{ };

	int count_accumulate{ };

	std::mutex grad_mutex{ };

};

} // namespace nn

#endif // NN_FULLY_CONNECTED_LAYER_H_