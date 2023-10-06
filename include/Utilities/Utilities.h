#ifndef NN_TRIVIAL_H_
#define NN_TRIVIAL_H_

using uint = long unsigned int;

namespace nn {
namespace util {

template<uint kFeatures>
using filter = std::array<std::array<float, kFeatures>, kFeatures>;


template<uint kFeatures, uint kChannels>
using image = std::array<filter<kFeatures>, kChannels>;


template<uint kFeatures, uint kChannels>
auto copy_to_vector (const image<kFeatures, kChannels>& X, std::vector<float>& V)
{
	V.clear(), V.reserve(kFeatures * kFeatures * kChannels);

	for (int f = 0; f < kChannels; f++)
		for (int i = 0; i < kFeatures; i++)
			for (int j = 0; j < kFeatures; j++)
				V.emplace_back(X[f][i][j]);
}


template<uint kFeatures, uint kChannels>
auto array_converted (const image<kFeatures, kChannels>& X)
{
	std::array<float, kFeatures * kFeatures * kChannels> Y{};
	for (int f = 0; f < kChannels; f++)
		for (int i = 0; i < kFeatures; i++)
			for (int j = 0; j < kFeatures; j++)
				Y[(f * kFeatures + i) * kFeatures + j] = X[f][i][j];
	return Y;
}


template<uint kFeatures, uint kChannels, uint kIn>
auto imagify (const std::array<float, kIn>& X)
{
	static_assert(kIn == kFeatures * kFeatures * kChannels);
	image<kFeatures, kChannels> Y{ };

	for (int f = 0; f < kChannels; f++)
		for (int i = 0; i < kFeatures; i++)
			for (int j = 0; j < kFeatures; j++)
				Y[f][i][j] = X[(f * kFeatures + i) * kFeatures + j];

	return Y;
}


template<uint kFeatures, uint kChannels>
auto imagify (const std::vector<float>& V)
{
	assert(kFeatures * kFeatures * kChannels == V.size());
	image<kFeatures, kChannels> X{ };

	for (int i = 0; i < kFeatures * kFeatures * kChannels; i++)
		X[i / (kFeatures * kFeatures)][(i / kFeatures) % kFeatures][i % kFeatures] = V[i];

	return X;
}


template<int kFeatures, int kChannels, int P>
auto pad (const image<kFeatures, kChannels>& X)
{
	image<kFeatures + P * 2, kChannels> Y{ };
	if constexpr (P == 0)
		Y = X;
	else if constexpr (P > 0) {
		for (int c = 0; c < kChannels; c++)
			for (int i = 0; i < kFeatures; i++)
				for (int j = 0; j < kFeatures; j++)
					Y[c][i + P][j + P] = X[c][i][j];
	} else {
		for (int c = 0; c < kChannels; c++)
			for (int i = P; i < kFeatures - P; i++)
				for (int j = P; j < kFeatures - P; j++)
					Y[c][i - P][j - P] = X[c][i][j];
	}
	return std::move(Y);
}


template<int kFeatures, int P>
auto pad (const filter<kFeatures>& X)
{
	filter<kFeatures + P * 2> Y{ };
	if constexpr (P == 0)
		Y = X;
	else if constexpr (P > 0) {
		for (int i = 0; i < kFeatures; i++)
			for (int j = 0; j < kFeatures; j++)
				Y[i + P][j + P] = X[i][j];
	} else {
		for (int i = P; i < kFeatures - P; i++)
			for (int j = P; j < kFeatures - P; j++)
				Y[i - P][j - P] = X[i][j];
	}
	return std::move(Y);
}

auto load_labeled_mnist (const std::string csv_path) {
	std::ifstream in(csv_path.c_str());

	constexpr int N = 28;

	std::vector<std::pair<nn::util::image<N, 1>, int>> set{};
	std::string s, word;

	while (in >> s) {
		std::vector<int> s_split{};
		s_split.reserve(N * N + 1);
		std::stringstream ss(s);

		while (!ss.eof())
			getline(ss, word, ','),
			s_split.push_back(stoi(word));

		if (s_split.size() != N * N + 1)
			std::cout << "Incorrect file format.\n", exit(0);

		nn::util::image<N, 1> img{};
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				img[0][i][j] = s_split[i * N + j + 1] / 256.0;

		auto &label = s_split[0];

		set.push_back(std::pair(img, label));
	}

	return std::move(set);
}

auto load_unlabeled_mnist (const std::string csv_path) {	
	std::ifstream in(csv_path.c_str());

	constexpr int N = 28;

	std::vector<nn::util::image<N, 1>> set{};
	std::string s, word;

	while (in >> s) {
		std::vector<int> s_split{};
		s_split.reserve(N * N);
		std::stringstream ss(s);

		while (!ss.eof())
			getline(ss, word, ','),
			s_split.push_back(stoi(word));

		if (s_split.size() != N * N)
			std::cout << "Incorrect file format.\n", exit(0);

		nn::util::image<N, 1> img{};

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				img[0][i][j] = s_split[i * N + j] / 256.0;

		set.push_back(img);
	}

	return std::move(set);
}


} // namespace util
} // namespace nn

#endif // NN_TRIVIAL_H_