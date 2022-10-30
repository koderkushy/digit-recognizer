#ifndef NN_LOSS_FUNCTIONS_CROSS_ENTROPY_H_
#define NN_LOSS_FUNCTIONS_CROSS_ENTROPY_H_

// namespace nn {
namespace LossFunctions {

template<typename T, uint64_t N>
auto soft_max (array<T, N> a) {
	T sum{};
	for (auto& x: a) sum += (x = exp(x));
	for (auto& x: a) x /= sum;
	return a;
}

struct CrossEntropy {
	template<typename T, uint64_t N>
	static auto loss (array<T, N> a, const int label) {
		a = soft_max(a);
		return -log(a[label] + 1e-5);
	}

	template<typename T, uint64_t N>
	static auto gradient (array<T, N> a, const int label) {
		a = soft_max(a);
		a[label] -= 1;
		return a;
	}
};

} // namespace lossfunctions
// } // namespace nn

#endif // NN_LOSS_FUNCTIONS_CROSS_ENTROPY_H_