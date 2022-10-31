#ifndef NN_LAYERS_OUTPUT_LAYER_H_
#define NN_LAYERS_OUTPUT_LAYER_H_


namespace nn {

template<
	int kClasses,
	class Loss
>
class OutputLayer {

public:

	OutputLayer ()
	{

	}


	template<uint64_t kFeatures, uint64_t kChannels>
	auto recurse (const nn::util::image<kFeatures, kChannels>& Y_img, int label)
	{
		static_assert(kFeatures == 1 and kChannels == kClasses);
		auto Y { nn::util::array_converted(Y_img) };
		return std::pair(nn::util::imagify<1, kClasses, kClasses>(Loss::gradient(Y, label)), Loss::loss(Y, label));
	}
	

	template<uint64_t kFeatures, uint64_t kChannels>
	auto evaluate (const nn::util::image<kFeatures, kChannels>& X, const int label)
	{
		static_assert(kFeatures == 1 and kChannels == kClasses);
		return Loss::loss(nn::util::array_converted(X), label);
	}


	void optimize ()
	{}


	void save (const std::string path)
	{}

};

} // namespace nn

#endif // NN_LAYERS_OUTPUT_LAYER_H_