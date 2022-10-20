namespace Optimizers {
	template<uint64_t N, uint64_t channels>
	struct RMSProp {
		const double rate, eps, decay;
		image<N, channels> MS_grad{};

		RMSProp (const double rate = 0.001, const double decay = 0.9, const double eps = 1e-7)
		: rate(rate), decay(decay), eps(eps) {}

		auto optimize (image<N, channels>& W, const image<N, channels>& grad_W) {
			for (int f = 0; f < channels; f++)
				for (int i = 0; i < N; i++)
					for (int j = 0; j < N; j++)
						MS_grad[f][i][j] = decay * MS_grad[f][i][j] + (1 - decay) * grad_W[f][i][j] * grad_W[f][i][j],
						W[f][i][j] += -rate * grad_W[f][i][j] / (eps + sqrt(MS_grad[f][i][j]));
		}
	};
}
