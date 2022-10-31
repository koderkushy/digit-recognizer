#ifndef NN_OPTIMIZERS_RMS_PROP_H_
#define NN_OPTIMIZERS_RMS_PROP_H_


// namespace nn {
namespace Optimizers {

	struct RMSProp {
		static double rate, eps, decay;
		double mean_square{};

		auto optimize (double& W, const double& G) {
			(mean_square *= decay) += (1 - decay) * G * G;
			W += -rate * G / (eps + sqrt(mean_square));
		}
	};

	double RMSProp::rate {0.01};
	double RMSProp::eps {1e-5};
	double RMSProp::decay {0.9};

} // namepsace optimizers
// } // namespace nn

#endif // NN_OPTIMIZERS_RMS_PROP_H_