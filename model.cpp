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
#include "ParametricReLU.h"
#include "FullyConnectedLayer.h"
#include "LossFunctions.h"
#include "Optimizers.h"


template<
	template<uint64_t N, uint64_t channels> class Optimizer,
	class Loss,
	int classes
>
struct model {
	ConvolutionalLayer<Optimizer, 1, 32, 5, 1> conv1;		// 28 -> 26
	ConvolutionalLayer<Optimizer, 32, 32, 5, 1> conv2;		// 26 -> 24
	ParametricReLU<Optimizer> relu1;
	MaxPool<24, 2, 0, 1> pool1;								// 24 -> 23
	DropOut drop1;
	ConvolutionalLayer<Optimizer, 32, 64, 3, 1> conv3;		// 23 -> 23
	ConvolutionalLayer<Optimizer, 64, 64, 3, 1> conv4;		// 23 -> 23
	ParametricReLU<Optimizer> relu2;
	MaxPool<23, 2, 0, 2> pool2;								// 23 -> 11
	DropOut drop2;
	FullyConnectedLayer<Optimizer, 11, 64, 32, 2> fcon1;
	ParametricReLU<Optimizer> relu3;
	DropOut drop3;
	FullyConnectedLayer<Optimizer, 32, 2, 1, classes> fcon2;

	mt19937 rng;

	using image28 = image<28, 1>;

	vector<pair<image28, int>> train_set, validation_set, test_set;

	model (): rng(chrono::high_resolution_clock::now().time_since_epoch().count()) {

	}

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
				img
			)))))))))));
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
				img
			)))), p[0]))))), p[1]))), p[2]));
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

	auto sgd (int epochs, int sample_size) {
		double best_validation_loss = std::numeric_limits<double>::max();
		assert(!train_set.empty() and !validation_set.empty());
		
		for (int i = 0; i < epochs; i++) {
			cout << "Epoch: " << i + 1 << endl;

			double training_loss{};
			array<double, classes> gradient{};

			for (int p = 0; p < sample_size; p++) {
				auto &[img, label] = train_set[rng() % train_set.size()];

				auto Y{forward_with_drop_out(img)};

				auto grad{Loss::gradient(array_converted(Y), label)};
				auto loss{Loss::loss(array_converted(Y), label)};

				training_loss += loss;
				for (int i = 0; i < classes; i++)
					gradient[i] += grad[i];
			}

			training_loss /= sample_size;
			for (int i = 0; i < classes; i++)
				gradient[i] /= sample_size;

			backward(gradient);

			double validation_loss{};
			for (auto [img, label]: validation_set) {
				validation_loss += Loss::loss(array_converted(forward(img)), label);
			}

			validation_loss /= sample_size;
			cout << "Validation loss = " << validation_loss << '\n';

			if (best_validation_loss > validation_loss)
				cout << "Previous best = " << best_validation_loss << '\n',
				cout << "Saving model...\n",
				best_validation_loss = validation_loss,
				save("model");

			cout << "=================================\n\n";
		}
	}

	auto load_data (const string csv_path) {
		if (freopen(csv_path.c_str(), "r", stdin) == NULL)
			cout << "Couldn't open file.\n", exit(0);

		constexpr int N = 28;

		vector<pair<image<N, 1>, int>> set{};
		string s, word; cin >> s;

		while (cin >> s) {
			vector<int> s_split{};
			s_split.reserve(N * N + 1);
			stringstream ss(s);

			while (!ss.eof())
				getline(ss, word, ','),
				s_split.push_back(stoi(word));

			if (s_split.size() != N * N + 1)
				cout << "Incorrect file format.\n", exit(0);

			image<N, 1> img{};
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					img[0][i][j] = s_split[i * N + j + 1] / 256.0;

			auto &label = s_split[0];

			set.push_back(pair(img, label));
		}

		return set;
	}

	auto train (const string train_csv_path, int epochs, int sample_size, double validation_ratio) {
		train_set = load_data(train_csv_path);

		shuffle(train_set.begin(), train_set.end(), rng);
		auto j = train_set.end() - train_set.size() * validation_ratio;
		
		validation_set = vector(j, train_set.end());
		train_set.erase(j, train_set.end());

		cout << "Training set has " << train_set.size() << " images\n"
			<< "Validation set has " << validation_set.size() << " images\n" << endl;

		sgd(epochs, sample_size);

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

    model<Optimizers::RMSProp, LossFunctions::CrossEntropy, 10> m;

    auto train_csv_path = "data/train/mnist_train_small.csv";

	cout << "Enter number of Epochs\n" << flush;
	int epoch_count; cin >> epoch_count;

	cout << "Enter sample size for gradient descent\n" << flush;
	int sample_size; cin >> sample_size;

	cout << "Enter fraction of training data to be used for validation\n" << flush;
	double validation_ratio; cin >> validation_ratio;

    m.train(train_csv_path, epoch_count, sample_size, validation_ratio);


}