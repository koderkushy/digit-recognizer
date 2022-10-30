#ifndef NN_LAYERS_OUTPUT_LAYER_H_
#define NN_LAYERS_OUTPUT_LAYER_H_

#include "../Operations/trivial.h"

namespace nn {

template<
	int classes,
	class Loss
>
struct OutputLayer {

	template<uint64_t N, uint64_t channels>
	auto recurse (const nn::operations::image<N, channels>& Y_img, int label) {
		static_assert(N == 1 and channels == classes);
		auto Y { nn::operations::array_converted(Y_img) };
		return pair(nn::operations::imagify<1, 10, 10>(Loss::gradient(Y, label)), Loss::loss(Y, label));
	}
	
	template<uint64_t N, uint64_t channels>
	auto evaluate (const nn::operations::image<N, channels>& X, const int label) {
		static_assert(N == 1 and channels == classes);
		return Loss::loss(nn::operations::array_converted(X), label);
	}

	void optimize () {}

	void save (const string path) {}
};

} // namespace nn

#endif // NN_LAYERS_OUTPUT_LAYER_H_