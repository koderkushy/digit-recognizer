#include "bits/stdc++.h"
using namespace std;


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
auto array_converted (const image<N, channels>& X) {
	array<double, N * N * channels> Y{};
	for (int f = 0; f < channels; f++)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				Y[(f * N + i) * N + j] = X[f][i][j];
	return Y;
}

template<uint64_t N, uint64_t channels, uint64_t W>
auto imagify (const array<double, W>& X) {
	static_assert(W == N * N * channels);
	image<N, channels> Y{};

	for (int f = 0; f < channels; f++)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				Y[f][i][j] = X[(f * N + i) * N + j];

	return Y;
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
	template<uint64_t N, uint64_t channels> class Optimizer,
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

	mt19937 rng;

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

	auto forward_with_drop_out (const image<28, 1>& img, const array<double, 3> p = {0.5, 0.5, 0.5}) {
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

	auto backward (const array<double, classes>& grad_Y) {
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
			imagify<1, classes, classes>(grad_Y)
		))))))))))))));
	}

	auto save (const string& path) {

	}

	template<int epochs, int sample_size>
	auto sgd (const vector<pair<image<28, 1>, int>>& training_set, const vector<pair<image<28, 1>, int>>& validation_set) {

		double best_validation_loss = std::numeric_limits<double>::max();

		for (int i = 0; i < epochs; i++) {
			double training_loss{};
			array<double, classes> gradient{};

			for (int p = 0; p < sample_size; p++) {
				auto &[img, label] = training_set[rng() % training_set.size()];

				auto Y{forward_with_drop_out(img)};

				auto grad{Loss::gradient(array_converted(Y), label)};
				auto loss{Loss::loss(array_converted(Y), label)};

				training_loss += loss;
				for (int i = 0; i < classes; i++)
					gradient[i] += grad[i];
			}

			for (int i = 0; i < classes; i++)
				gradient[i] /= sample_size;

			backward(gradient);

			double validation_loss{};
			for (auto [img, label]: validation_set) {
				validation_loss += Loss::loss(array_converted(forward(img)), label);
			}

			validation_loss /= sample_size;

			if (best_validation_loss > validation_loss)
				cout << "New loss = " << validation_loss << '\n',
				cout << "Previous best = " << best_validation_loss << '\n',
				cout << "Saving model...\n",
				best_validation_loss = validation_loss,
				save("model");
		}
	}

	model (const double relu_slope): relu1(relu_slope), relu2(relu_slope), relu3(relu_slope), rng(chrono::high_resolution_clock::now().time_since_epoch().count()) {}

	auto train (vector<pair<image<28, 1>, int>> training_set) {
		shuffle(training_set.begin(), training_set.end(), rng);
		vector validation_set(training_set.end() - training_set.size() * 0.2, training_set.end());

		sgd<100, 64>(training_set, validation_set);
	}

	auto test (const vector<pair<image<28, 1>, int>>& test_set) {
		int correct_guesses{};

		for (auto [img, label]: test_set) {
			auto Y{forward(img)};
			bool correct = true;

			for (int j = 0; j < classes; j++)
				if (Y[j][0][0] > Y[label][0][0] + 1e-7)
					correct = false;
			correct_guesses += correct;
		}

		cout << "Accuracy = " << correct_guesses * 100.0 / test_set.size() << '\n';
	}
};

int main(){
    ios_base::sync_with_stdio(0), cin.tie(0);

    
    vector training_set(0, pair<image<28, 1>, int>());
    vector test_set(0, pair<image<28, 1>, int>());

    model<Optimizers::RMSProp, LossFunctions::CrossEntropy, 10> m(0.01);

    m.train(training_set);
    m.test(test_set);

}