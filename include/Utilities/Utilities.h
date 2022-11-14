#ifndef NN_TRIVIAL_H_
#define NN_TRIVIAL_H_


namespace nn {
namespace util {

template<int kFeatures>
using filter = std::array<std::array<float, kFeatures>, kFeatures>;


template<int kFeatures, int kChannels>
using image = std::array<filter<kFeatures>, kChannels>;


template<uint64_t kFeatures, uint64_t kChannels>
auto copy_to_vector (const image<kFeatures, kChannels>& X, std::vector<float>& V)
{
	V.clear(), V.reserve(kFeatures * kFeatures * kChannels);

	for (int f = 0; f < kChannels; f++)
		for (int i = 0; i < kFeatures; i++)
			for (int j = 0; j < kFeatures; j++)
				V.emplace_back(X[f][i][j]);
}


template<uint64_t kFeatures, uint64_t kChannels>
auto array_converted (const image<kFeatures, kChannels>& X)
{
	std::array<float, kFeatures * kFeatures * kChannels> Y{};
	for (int f = 0; f < kChannels; f++)
		for (int i = 0; i < kFeatures; i++)
			for (int j = 0; j < kFeatures; j++)
				Y[(f * kFeatures + i) * kFeatures + j] = X[f][i][j];
	return Y;
}


template<uint64_t kFeatures, uint64_t kChannels, uint64_t kIn>
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


template<uint64_t kFeatures, uint64_t kChannels>
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


} // namespace util
} // namespace nn

#endif // NN_TRIVIAL_H_