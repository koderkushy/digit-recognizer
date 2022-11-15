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


	auto recurse (const nn::util::image<1, kClasses>& Y_img, int label)
	{
		auto Y { nn::util::array_converted(Y_img) };
		return std::pair(nn::util::imagify<1, kClasses, kClasses>(Loss::gradient(Y, label)), Loss::loss(Y, label));
	}
	

	auto evaluate (const nn::util::image<1, kClasses>& X, const int label)
	{
		return Loss::loss(nn::util::array_converted(X), label);
	}


	auto predict (const nn::util::image<1, kClasses>& X)
	{
		return max_element(X.begin(), X.end()) - X.begin();
	}


	void optimize ()
	{}


	void save (const std::string path)
	{}

};

} // namespace nn

#endif // NN_LAYERS_OUTPUT_LAYER_H_