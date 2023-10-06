#ifndef NN_FULLY_CONNECTED_LAYER_H_
#define NN_FULLY_CONNECTED_LAYER_H_


namespace nn {

template<
	int kFeatures,
	int kChannels,
	int kOutWidth,
	class Optimizer,
	class NextLayer
>
class FullyConnected {
	static constexpr int kInWidth = kFeatures * kFeatures * kChannels;

public:

	FullyConnected ()
	{
		// Kaiming He initialisation

		std::mt19937 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		std::normal_distribution<float>
			gaussian { 0, sqrt(1.0 / kInWidth) };

		for (int i = 0; i < kInWidth; i++)
			for (int j = 0; j < kOutWidth; j++)
				W[i][j] = gaussian(rng);

		for (int i = 0; i < kOutWidth; i++)
			b[i] = gaussian(rng);
	}


	auto recurse (const nn::util::image<kFeatures, kChannels>& X, const int label)
	{
		const auto [gradient, loss] = L.recurse(forward(X), label);
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

		out.close();
		
		L.save(path);
	}


private:

	auto forward (const nn::util::image<kFeatures, kChannels>& X) const
	{

			// auto start = std::chrono::high_resolution_clock::now();

		auto arr_X { nn::util::array_converted(X) };
		std::array<float, kOutWidth> arr_Y { b };

		for (int i = 0; i < kInWidth; i++)
#pragma GCC ivdep
			for (int j = 0; j < kOutWidth; j++)
				arr_Y[j] += W[i][j] * arr_X[i];

			// auto stop = std::chrono::high_resolution_clock::now();
			// std::cout << "fcon fwd = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms\n" << std::flush;

		return nn::util::imagify<1, kOutWidth, kOutWidth>(arr_Y);
	}


	auto backward (const nn::util::image<kFeatures, kChannels>& X, const nn::util::image<1, kOutWidth>& grad_Y)
	{

			// auto start = std::chrono::high_resolution_clock::now();

		auto arr_grad_Y { nn::util::array_converted(grad_Y) };
		auto arr_X { nn::util::array_converted(X) };

		std::array<float, kInWidth> arr_grad_X { };

		const auto& grad_b { arr_grad_Y };

		{
			int i = 0;
			for (; i + 4 <= kInWidth; i += 4) {
#pragma GCC ivdep
				for (int j = 0; j < kOutWidth; j++) {
					arr_grad_X[i + 0] += arr_grad_Y[j] * W[i + 0][j];
					arr_grad_X[i + 1] += arr_grad_Y[j] * W[i + 1][j];
					arr_grad_X[i + 2] += arr_grad_Y[j] * W[i + 2][j];
					arr_grad_X[i + 3] += arr_grad_Y[j] * W[i + 3][j];
				}
			}

			for (; i < kInWidth; i++)
#pragma GCC ivdep
				for (int j = 0; j < kOutWidth; j++)
					arr_grad_X[i] += arr_grad_Y[j] * W[i][j];
		}

		{
			std::lock_guard<std::mutex> lock(grad_mutex);

			count_accumulate++;

#pragma GCC ivdep
			for (int i = 0; i < kOutWidth; i++)
				grad_b_accumulate[i] += grad_b[i];

			{
				int i = 0;
				for (; i + 4 <= kInWidth; i += 4) {
#pragma GCC ivdep
					for (int j = 0; j < kOutWidth; j++) {
						grad_W_accumulate[i + 0][j] += arr_grad_Y[j] * arr_X[i + 0];
						grad_W_accumulate[i + 1][j] += arr_grad_Y[j] * arr_X[i + 1];
						grad_W_accumulate[i + 2][j] += arr_grad_Y[j] * arr_X[i + 2];
						grad_W_accumulate[i + 3][j] += arr_grad_Y[j] * arr_X[i + 3];
					}
				}

				for (; i < kInWidth; i++)
#pragma GCC ivdep
					for (int j = 0; j < kOutWidth; j++)
						grad_W_accumulate[i][j] += arr_grad_Y[j] * arr_X[i];
			}
		}

			// auto stop = std::chrono::high_resolution_clock::now();

			// std::cout << "fcon bwd = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms\n" << std::flush;

		return nn::util::imagify<kFeatures, kChannels, kInWidth>(arr_grad_X);
	}


	std::array<std::array<float, kOutWidth>, kInWidth> W{ };

	std::array<float, kOutWidth> b{ };

	NextLayer L{ };

	std::array<std::array<float, kOutWidth>, kInWidth> grad_W_accumulate{ };

	std::array<float, kOutWidth> grad_b_accumulate{ };

	int count_accumulate{ };

	std::mutex grad_mutex{ };

};

} // namespace nn

#endif // NN_FULLY_CONNECTED_LAYER_H_