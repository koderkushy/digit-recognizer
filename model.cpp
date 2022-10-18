#include "bits/stdc++.h"
using namespace std;

constexpr double eta = 0.1;

template<int N>
using filter = array<array<double, N>, N>;

template<int N, int C>
using image = array<filter<N>, C>;

template<typename T, typename U>
T min (const T& x, const U& y) { return std::min(x, static_cast<T>(y)); }
template<typename T, typename U>
T max (const T& x, const U& y) { return std::max(x, static_cast<T>(y)); }

template<int N, int channels>
auto copy_to_vector (const image<N, channels>& X, vector<vector<vector<double>>>& V) {
	V = vector(channels, vector(N, vector(N, double())));

	for (int f = 0; f < channels; f++)
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					V[f][i][j] = X[f][i][j];
}


#include "ConvolutionalLayer.h"
#include "MaxPooling.h"
#include "DropOut.h"
#include "ParametricRelU.h"
#include "FullyConnectedLayer.h"

template<int classes>
struct model {
	ConvolutionalLayer<1, 32, 5, 1> conv1;
	ConvolutionalLayer<32, 32, 5, 1> conv2;
	ConvolutionalLayer<32, 64, 3, 1> conv3;
	ConvolutionalLayer<64, 64, 3, 1> conv4;
	FullyConnectedLayer<22, 64, 32, 2> fcon1;
	FullyConnectedLayer<32, 2, 1, classes> fcon2;
	MaxPool<2, 0> pool1;
	MaxPool<2, 0> pool2;
	ParametricReLU relu1, relu2, relu3;
	DropOut drop1, drop2, drop3;

	auto evaluate (const image<28, 1>& img) {
		return
			fcon2.evaluate(
			relu3.evaluate(
			fcon1.evaluate(
			pool2.evaluate(
		  	relu2.evaluate(
		  	conv4.evaluate(
		  	conv3.evaluate(
			pool1.evaluate(
			relu1.evaluate(
			conv2.evaluate(
			conv1.evaluate(
				img)))))))))));
	}

	auto evaluate_with_drop_out (const image<28, 1>& img, const array<double, 3> p) {
		return
			fcon2.evaluate(
			drop3.evaluate(
			relu3.evaluate(
			fcon1.evaluate(
			drop2.evaluate(
			pool2.evaluate(
		  	relu2.evaluate(
		  	conv4.evaluate(
		  	conv3.evaluate(
		  	drop1.evaluate(
			pool1.evaluate(
			relu1.evaluate(
			conv2.evaluate(
			conv1.evaluate(
				img)))), p[0]))))), p[1]))), p[2]));
	}

	auto back_propagate (const image<1, classes>& grad_Y) {
		conv1.back_propagate(
		conv2.back_propagate(
		relu1.back_propagate(
		pool1.back_propagate(
		drop1.back_propagate(
		conv3.back_propagate(
		conv4.back_propagate(
		relu2.back_propagate(
		pool2.back_propagate(
		drop2.back_propagate(
		fcon1.back_propagate(
		relu3.back_propagate(
		drop3.back_propagate(
		fcon2.back_propagate(
			grad_Y))))))))))))));
	}

	bool is_good_sample (const vector<int>& labels) {
		array<int, classes> counts{};
		for (int x: labels)
			counts[x]++;
		for (int c: counts)
			if (c > labels.size() / 2)
				return false;
		return true;
	}

	model () {

	}

	template<int epochs, int sample_size>
	auto gradient_descent (const pair<vector<image<28, 1>>, int>& T) {

	}

	auto train (const pair<vector<image<28, 1>>, int>& T) {
		
	}
};

int main(){
    ios_base::sync_with_stdio(0), cin.tie(0);

    
    vector training_set(0, pair<image<28, 1>, int>());





}