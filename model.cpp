#include "bits/stdc++.h"
using namespace std;

constexpr double eta = 0.1;

template<int N> using filter = array<array<double, N>, N>;
template<int N, int C> using image = array<filter<N>, C>;
template<typename T, typename U> T min (const T& x, const U& y) { return std::min(x, static_cast<T>(y)); }
template<typename T, typename U> T max (const T& x, const U& y) { return std::max(x, static_cast<T>(y)); }

template<uint64_t N, uint64_t channels>
auto copy_to_vector (const image<N, channels>& X, vector<double>& V) {
	V.clear(), V.reserve(N * N * channels);

	for (int f = 0; f < channels; f++)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				V.emplace_back(X[f][i][j]);
}

template<uint64_t N, uint64_t channels>
auto imagify (const vector<double>& V) {
	image<N, channels> X{};

	for (int f = 0; f < channels; f++)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				X[f][i][j] = V[(f * N + i) * N + j];

	return X;
}

#include "ConvolutionalLayer.h"
#include "MaxPooling.h"
#include "DropOut.h"
#include "ParametricRelU.h"
#include "FullyConnectedLayer.h"
#include "LossFunctions.h"
#include "Optimizers.h"


template<
	template<int N, int channels> class Optimizer,
	class Loss,
	int classes
>
struct model {
	ConvolutionalLayer<Optimizer, 1, 32, 5, 1> conv1;
	ConvolutionalLayer<Optimizer, 32, 32, 5, 1> conv2;
	ConvolutionalLayer<Optimizer, 32, 64, 3, 1> conv3;
	ConvolutionalLayer<Optimizer, 64, 64, 3, 1> conv4;
	FullyConnectedLayer<Optimizer, 22, 64, 32, 2> fcon1;
	FullyConnectedLayer<Optimizer, 32, 2, 1, classes> fcon2;
	MaxPool<2, 0> pool1;
	MaxPool<2, 0> pool2;
	ParametricReLU<Optimizer> relu1, relu2, relu3;
	DropOut drop1, drop2, drop3;


	auto forward (const image<28, 1>& img) {
		return
			fcon2.forward(
			relu3.forward(
			fcon1.forward(
			pool2.forward(
		  	relu2.forward(
		  	conv4.forward(
		  	conv3.forward(
			pool1.forward(
			relu1.forward(
			conv2.forward(
			conv1.forward(
				img)))))))))));
	}

	auto forward_with_drop_out (const image<28, 1>& img, const array<double, 3> p) {
		return
			fcon2.train(
			drop3.train(
			relu3.train(
			fcon1.train(
			drop2.train(
			pool2.train(
		  	relu2.train(
		  	conv4.train(
		  	conv3.train(
		  	drop1.train(
			pool1.train(
			relu1.train(
			conv2.train(
			conv1.train(
				img)))), p[0]))))), p[1]))), p[2]));
	}

	auto backward (const image<1, classes>& grad_Y) {
		conv1.backward(
		conv2.backward(
		relu1.backward(
		pool1.backward(
		drop1.backward(
		conv3.backward(
		conv4.backward(
		relu2.backward(
		pool2.backward(
		drop2.backward(
		fcon1.backward(
		relu3.backward(
		drop3.backward(
		fcon2.backward(
			grad_Y))))))))))))));
	}

	template<int epochs, int sample_size>
	auto sgd (const vector<pair<image<28, 1>, int>>& T, const double persistence) {
		assert(0 < persistence and persistence < 1);

		mt19937 rng (chrono::high_resolution_clock::now().time_since_epoch().count());

		auto is_good = [&](const vector<int>& labels) {
			array<int, classes> counts{};
			for (int x: labels)
				counts[x]++;
			for (int c: counts)
				if (c > labels.size() / 2)
					return false;
			return true;
		};
		auto generate_sample = [&]() {
			vector<int> v(sample_size), labels(sample_size);
			do {
				for (int i = 0; i < sample_size; i++) {
					int j = rng() % T.size();
					v[i] = j, labels[i] = j;
				}
			} while (!is_good(labels));
			return v;
		};
		auto loss = [&](const array<double, classes>& Y, const int label) {
			double loss{};

			for (int j = 0; j < classes; j++) {
				if (j == label) loss += exp(Y[j]);
				else loss += exp(-Y[j]);
			}

			return loss;
		};
		auto gradient = [&](const array<double, classes>& Y, const int label) {
			array<double, classes> grad{};

			for (int j = 0; j < classes; j++) {
				if (j == label) grad[j] += exp(Y[j]);
				else grad[j] += -exp(-Y[j]);
			}

			return grad;
		};

		array<double, classes> prev{};

		for (int i = 0; i < epochs; i++) {
			
			array<double, classes> grad{};
			double training_loss{};

			for (int i: generate_sample()) {
				auto &[data, label] = T[i];
				auto Y = forward_with_drop_out(data);
				for (int j = 0; j < classes; j++) {
					if (j == label)
						grad[j] += exp(Y[j]), training_loss += exp(Y[j]);
					else
						grad[j] += -exp(-Y[j]), training_loss += exp(-Y[j]);
				}
			}

			for (auto& g: grad) g /= sample_size;
			training_loss /= sample_size;

			for (int j = 0; j < classes; j++)
				prev[j] = grad[j] + prev[j] * persistence;

			backward(prev);
		}


	}

	model (const double relu_slope): relu1(relu_slope), relu2(relu_slope), relu3(relu_slope) {}

	auto train (const vector<pair<image<28, 1>, int>>& T) {
		sgd<100, 32>(T, 0.2);
	}
};

int main(){
    ios_base::sync_with_stdio(0), cin.tie(0);

    
    vector training_set(0, pair<image<28, 1>, int>());

    model<Optimizers::RMSProp, LossFunctions::CrossEntropy, 10> m(0.01);



}