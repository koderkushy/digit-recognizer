#ifndef NN_OPTIMIZERS_RMS_PROP_H_
#define NN_OPTIMIZERS_RMS_PROP_H_


namespace nn {
namespace Optimizers {

	template<class Params>
	class RmsProp {
		static constexpr float decay = Params::decay;
		static constexpr float rate = Params::rate;
		static constexpr float eps = Params::eps;

		float mean_square{ };

	public:
		auto optimize (float& W, const float& G) {
			(mean_square *= decay) += (1 - decay) * G * G;
			W += -rate * G / (eps + sqrt(mean_square));
		}
	};


	template<class Params>
	class GdMomentum {
		static constexpr float decay = Params::decay;
		static constexpr float rate = Params::rate;

		float d_W{ };

	public:
		auto optimize (float& W, const float& G) {
			(d_W *= decay) += (1 - decay) * G;
			W += -rate * d_W;
		}
	};


} // namepsace optimizers
} // namespace nn

#endif // NN_OPTIMIZERS_RMS_PROP_H_