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
		const auto Y { nn::util::array_converted(Y_img) };
		return std::pair(nn::util::imagify<1, kClasses, kClasses>(Loss::gradient(Y, label)), Loss::loss(Y, label));
	}
	

	auto evaluate (const nn::util::image<1, kClasses>& X, const int label) const
	{
		return Loss::loss(nn::util::array_converted(X), label);
	}


	int predict (const nn::util::image<1, kClasses>& X_img) const
	{
		const auto X { nn::util::array_converted(X_img) };
		return std::max_element(X.begin(), X.end()) - X.begin();
	}


	void optimize ()
	{}


	void save (const std::string path, const int layer_index = 0) const
	{
		std::ofstream desc_out(path + "model_description.txt"
									, std::ios::out | std::ios::app);
		std::ofstream lyr_stream(path + "layer-" + std::to_string(layer_index) + ".bin"
									, std::ios::out | std::ios::binary);

		desc_out << "output\n"
			<< "classes " << kClasses << '\n' << std::flush;
		
		desc_out.close();
		lyr_stream.close();
	}

	void load (const std::string path, const int layer_index = 0)
	{
		std::ifstream lyr_stream(path + "layer-" + std::to_string(layer_index) + ".bin"
									, std::ios::in | std::ios::binary);
		
		assert(lyr_stream.is_open());
		lyr_stream.close();
	}

	size_t size () const {
		return 0;
	}

};

} // namespace nn

#endif // NN_LAYERS_OUTPUT_LAYER_H_