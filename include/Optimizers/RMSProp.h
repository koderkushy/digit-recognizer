#ifndef NN_OPTIMIZERS_RMS_PROP_H_
#define NN_OPTIMIZERS_RMS_PROP_H_


// namespace nn {
namespace Optimizers {

	struct RMSProp {
		static float rate, eps, decay;
		float mean_square{};

		auto optimize (float& W, const float& G) {
			(mean_square *= decay) += (1 - decay) * G * G;
			W += -rate * G / (eps + sqrt(mean_square));
		}
	};

	float RMSProp::rate {0.01};
	float RMSProp::eps {1e-5};
	float RMSProp::decay {0.9};

} // namepsace optimizers
// } // namespace nn

#endif // NN_OPTIMIZERS_RMS_PROP_H_