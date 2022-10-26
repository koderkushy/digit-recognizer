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
	double RMSProp::eps {1e-8};
	double RMSProp::decay {0.9};

}
