namespace LossFunctions {
	template<typename T, int N>
	auto soft_max (array<T, N> a) {
		T sum{};
		for (auto& x: a) sum += (x = exp(x));
		for (auto& x: a) x /= sum;
		return a;
	}

	struct CrossEntropy {
		template<typename T, int N>
		static auto loss (array<T, N> a, const int label) {
			a = soft_max(a);
			return -log(a[label]);
		}

		template<typename T, int N>
		static auto gradient (array<T, N> a, const int label) {
			a = soft_max(a);
			a[label] -= 1;
			return a;
		}
	};
}
