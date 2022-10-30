#ifndef NN_DROP_OUT_H_
#define NN_DROP_OUT_H_

#include "../Operations/trivial.h"

namespace nn {

template<
	int percent,
	class NextLayer
>
struct DropOut {
	static_assert(0 <= percent and percent < 100);

	mt19937 rng;
	static constexpr uint64_t R = std::numeric_limits<uint32_t>::max();

	NextLayer L { };

	DropOut (): rng(chrono::high_resolution_clock::now().time_since_epoch().count()) {}

	template<uint64_t N, uint64_t channels>
	auto recurse (nn::operations::image<N, channels> X, const int label) {
		array<bool, N * N * channels> mask { };

		for (auto& b: mask)
			b = (rng() > percent * R / 100.0);

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					X[f][i][j] *= mask[(f * N + i) * N + j];

		auto [gradient, loss] = L.recurse(X, label);

		for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					gradient[f][i][j] *= mask[(f * N + i) * N + j];

		return pair(std::move(gradient), loss);

	}

	template<uint64_t N, uint64_t channels>
	auto evaluate (const nn::operations::image<N, channels>& X, const int label) {
		return L.evaluate(X, label);
	}

	auto save (const string path) {
		L.save(path);
	}

	auto optimize () {
		L.optimize();
	}
};

} // namespace nn

#endif // NN_DROP_OUT_H_