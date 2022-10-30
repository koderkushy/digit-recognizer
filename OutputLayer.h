#ifndef NN_LAYERS_OUTPUT_LAYER_H_
#define NN_LAYERS_OUTPUT_LAYER_H_

// namespace nn {

template<
	int classes,
	class Loss
>
struct OutputLayer {

	template<uint64_t N, uint64_t channels>
	auto recurse (const image<N, channels>& Y_img, int label) {
		static_assert(N == 1 and channels == classes);
		auto Y { array_converted(Y_img) };
		return pair(imagify<1, 10, 10>(Loss::gradient(Y, label)), Loss::loss(Y, label));
	}
	
	template<uint64_t N, uint64_t channels>
	auto evaluate (const image<N, channels>& X, const int label) {
		static_assert(N == 1 and channels == classes);
		return Loss::loss(array_converted(X), label);
	}

	void optimize () {}

	void save (const string path) {}
};

// } // namespace nn

#endif // NN_LAYERS_OUTPUT_LAYER_H_