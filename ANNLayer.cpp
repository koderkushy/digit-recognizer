#include "bits/stdc++.h"
using namespace std;

constexpr double eta = 0.1;

template<int L, int R>
struct ANNLayer {

	array<array<double, L>, R> w;
	array<double, R> b;
	array<double, L> last_input;

	template<bool training = 0>
	auto evaluate (const array<double, L>& x) {
		array<double, R> y;
		if constexpr (training)
			last_input = x;

		for (int i = 0; i < R; i++)
			for (int j = 0; j < L; j++)
				y[i] += x[j] * w[j][i] + b[i];

		return y;
	}

	auto back_propagate (const array<double, R>& grad_y) {
		array<double, L> grad_x;

		for (int i = 0; i < L; i++)
			for (int j = 0; j < R; j++)
				grad_x[i] += grad_y[j] * w[i][j],
				w[i][j] += - eta * grad_y[j] * last_input[i];

		return grad_x;
	}

};

int main(){
    ios_base::sync_with_stdio(0), cin.tie(0);

    



}