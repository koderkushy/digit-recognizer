#include "bits/stdc++.h"
#include <execution>
using namespace std;


namespace cnn {

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
		assert(N * N * channels == V.size());
		image<N, channels> X{};

		for (int i = 0; i < N * N * channels; i++)
			X[i / (N * N)][(i / N) % N][i % N] = V[i];

		return X;
	}

	template<int N, int C, int P>
	auto pad (const image<N, C>& X) {
		image<N + P * 2, C> Y{};
		if constexpr (P == 0)
			Y = X;
		else if constexpr (P > 0) {
			for (int c = 0; c < C; c++)
				for (int i = 0; i < N; i++)
					for (int j = 0; j < N; j++)
						Y[c][i + P][j + P] = X[c][i][j];
		} else {
			for (int c = 0; c < C; c++)
				for (int i = P; i < N - P; i++)
					for (int j = P; j < N - P; j++)
						Y[c][i - P][j - P] = X[c][i][j];
		}
		return std::move(Y);
	}

	template<int N, int P>
	auto pad (const filter<N>& X) {
		filter<N + P * 2> Y{};
		if constexpr (P == 0)
			Y = X;
		else if constexpr (P > 0) {
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					Y[i + P][j + P] = X[i][j];
		} else {
			for (int i = P; i < N - P; i++)
				for (int j = P; j < N - P; j++)
					Y[i - P][j - P] = X[i][j];
		}
		return std::move(Y);
	}

	template<uint64_t M, uint64_t N, uint64_t K>
	auto mat_mul (const array<array<double, N>, M>& A, const array<array<double, K>, N>& B) {
		array<array<double, K>, M> C {};

		for (int i = 0; i < M; i++)
				for (int k = 0; k < N; k++)
			for (int j = 0; j < K; j++)
					C[i][j] += A[i][k] * B[k][j];

		return std::move(C);
	}

	template<uint64_t N, uint64_t NC, uint64_t K, uint64_t KC>
	auto convolve (const image<N, NC>& X, const array<image<K, NC>, KC>& W) {
		static constexpr int M = N - K + 1;
		image<M, KC> Y{};

		array<array<double, K * K * NC>, KC> W_mat{};
		for (int f = 0; f < KC; f++)
			for (int g = 0; g < NC; g++)
				for (int i = 0; i < K; i++)
					for (int j = 0; j < K; j++)
						W_mat[f][(g * K + i) * K + j] = W[f][g][i][j];

		array<array<double, M * M>, K * K * NC> X_mat{};	
		for (int i = 0; i < M; i++)
			for (int j = 0; j < M; j++)
				for (int g = 0; g < NC; g++)
					for (int x = 0; x < K; x++)
						for (int y = 0; y < K; y++)
							X_mat[y + K * (x + K * g)][i * M + j] = X[g][i + x][j + y];

		auto Y_mat {mat_mul(W_mat, X_mat)};

		for (int f = 0; f < KC; f++)
			Y[f] = imagify<M, 1, M * M>(Y_mat[f])[0];

		return std::move(Y);
	}


	template<uint64_t N, uint64_t K>
	auto convolve (const filter<N>& X, const filter<K>& W) {
		image<N, 1> _X{}; _X[0] = X;
		array<image<K, 1>, 1> _W{}; _W[0][0] = W;
		return convolve(_X, _W);
	}


}

using namespace cnn;

#include "ConvolutionalLayer.h"
#include "MaxPooling.h"
#include "DropOut.h"
#include "ParametricReLU.h"
#include "FullyConnectedLayer.h"
#include "LossFunctions.h"
#include "Optimizers.h"
#include "ReLU.h"


template<
	class OptimizerClass,
	class Loss,
	int classes
>
struct model {
	ConvolutionalLayer<OptimizerClass, 1, 32, 5, 1> conv1;		// 28 -> 26
	ConvolutionalLayer<OptimizerClass, 32, 32, 5, 1> conv2;		// 26 -> 24
	ReLU relu1;
	MaxPool<24, 2, 0, 1> pool1;								// 24 -> 23
	DropOut drop1;
	ConvolutionalLayer<OptimizerClass, 32, 64, 3, 1> conv3;		// 23 -> 23
	ConvolutionalLayer<OptimizerClass, 64, 64, 3, 1> conv4;		// 23 -> 23
	ReLU relu2;
	MaxPool<23, 2, 0, 2> pool2;								// 23 -> 11
	DropOut drop2;
	FullyConnectedLayer<OptimizerClass, 11, 64, 32, 2> fcon1;
	ReLU relu3;
	DropOut drop3;
	FullyConnectedLayer<OptimizerClass, 32, 2, 1, classes> fcon2;

	mt19937 rng;

	using image28 = image<28, 1>;

	vector<pair<image28, int>> train_set, validation_set, test_set;

	model (): rng(chrono::high_resolution_clock::now().time_since_epoch().count()) {
		cout << "\nInitialised all layers.\n\n" << flush;
	}

	template<bool with_drop_out = false>
	auto forward (const image<28, 1>& img, const array<double, 3> p = {0.5, 0.5, 0.5}) {
		if constexpr (with_drop_out)
			return fcon2.train(drop3.train(relu3.train(fcon1.train(drop2.train(pool2.train(relu2.train(conv4.train(conv3.train(drop1.train(pool1.train(relu1.train(conv2.train(conv1.train(img)))), p[0]))))), p[1]))), p[2]));
		else
			return fcon2.forward(relu3.forward(fcon1.forward(pool2.forward(relu2.forward(conv4.forward(conv3.forward(pool1.forward(relu1.forward(conv2.forward(conv1.forward(img)))))))))));
	}
	
	template<bool with_drop_out = false>
	auto forward (const vector<pair<image<28, 1>, int>>& set, const array<double, 3> p = {0.5, 0.5, 0.5}) {
		vector<pair<array<double, classes>, int>> res(set.size());
		vector<int> index(set.size());
		iota(begin(index), end(index), 0);

		for_each (execution::par, begin(index), end(index), [&](const auto& i) {
			const auto& [img, label] = set[i];
			res[i] = pair(array_converted(forward<with_drop_out>(img, p)), label);
		});

		return std::move(res);
	}

	auto backward (const array<double, classes>& grad_Y) {
		conv1.backward(conv2.backward(relu1.backward(pool1.backward(drop1.backward(conv3.backward(conv4.backward(relu2.backward(pool2.backward(drop2.backward(fcon1.backward(relu3.backward(drop3.backward(fcon2.backward(imagify<1, classes, classes>(grad_Y)))))))))))))));
	}

	auto save (const string& path) {
		conv1.save(path + "/conv1.csv");
		conv2.save(path + "/conv2.csv");
		conv3.save(path + "/conv3.csv");
		conv4.save(path + "/conv4.csv");
		fcon1.save(path + "/fcon1.csv");
		fcon2.save(path + "/fcon2.csv");
		// relu1.save(path + "/relu1.csv");
		// relu2.save(path + "/relu2.csv");
		// relu3.save(path + "/relu3.csv");
	}

	auto sgd (int epochs, int sample_size) {
		double best_validation_loss = std::numeric_limits<double>::max();
		assert(!train_set.empty() and !validation_set.empty());
		
		for (int i = 1; i < epochs + 1; i++) {
			cout << "Epoch: " << i << '\n' << flush;

			vector<pair<image28, int>> sample_set(sample_size);

			generate (begin(sample_set), end(sample_set), [&]() {
				return train_set[rng() % train_set.size()];
			});

			array<double, classes> gradient{};

			auto start = chrono::high_resolution_clock::now();
			double training_loss {};

			for (const auto& [x, label]: forward<true>(sample_set)) {
				auto grad {Loss::gradient(x, label)};
				training_loss += Loss::loss(x, label);
				for (int i = 0; i < classes; i++)
					gradient[i] += grad[i];
			}

			auto stop = chrono::high_resolution_clock::now();

			cout << "Time = " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << "ms\n";

			cout << "Gradients = [" << fixed << setprecision(4);
			for (int i = 0; i < classes; i++)
				cout << (gradient[i] /= sample_size) << ' ' ;
			cout << "]\nLoss = " << training_loss / sample_size << "\n\n";

			backward(gradient);

			if (i % 25 == 0) {
				cout << "Validating...\n" << flush;

				double validation_loss{};

				for (const auto& [x, label]: forward<false>(validation_set)) {
					validation_loss += Loss::loss(x, label);
				}

				validation_loss /= validation_set.size();
				cout << "Validation loss = " << validation_loss << '\n';

				if (best_validation_loss > validation_loss)
					cout << "Previous best = " << best_validation_loss << '\n',
					cout << "Saving model...\n",
					best_validation_loss = validation_loss,
					save("model_parameters");
			}

			cout << "=================================\n\n";
		}
	}

	auto load_data (const string csv_path) {
		if (freopen(csv_path.c_str(), "r", stdin) == NULL)
			cout << "Couldn't open file.\n", exit(0);

		constexpr int N = 28;

		vector<pair<image<N, 1>, int>> set{};
		string s, word;
		// cin >> s;

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

		return std::move(set);
	}

	auto train_all (const int epochs, const array<double, 3> dropout_ratios = {0.5, 0.5, 0.5}) {
		
		double best_validation_loss { std::numeric_limits<double>::max() };

		for (int i = 0; i < epochs; i++) {
			cout << "Epoch: " << i + 1 << endl;

			shuffle(begin(train_set), end(train_set), rng);

			double training_loss { };

			for (const auto& [img, label]: train_set) {
				auto confi { array_converted(forward<true>(img, dropout_ratios))};

				backward(Loss::gradient(confi, label));
				training_loss += Loss::loss(confi, label);
			}

			cout << "Training loss = " << training_loss << endl;
			cout << "Validating...\n" << flush;

			double validation_loss{};

			for (const auto& [x, label]: forward<false>(validation_set)) {
				validation_loss += Loss::loss(x, label);
			}

			validation_loss /= validation_set.size();
			cout << "Validation loss = " << validation_loss << '\n';

			if (best_validation_loss > validation_loss)
				cout << "Previous best = " << best_validation_loss << '\n',
				cout << "Saving model...\n",
				best_validation_loss = validation_loss,
				save("model_parameters");
		}
	}

	auto train (const string train_csv_path, int epochs, int sample_size, double validation_ratio) {
		train_set = load_data(train_csv_path);

		shuffle(train_set.begin(), train_set.end(), rng);
		auto j = train_set.end() - train_set.size() * validation_ratio;
		
		validation_set = vector(j, train_set.end());
		train_set.erase(j, train_set.end());

		cout << "Training set has " << train_set.size() << " images\n"
			<< "Validation set has " << validation_set.size() << " images\n" << endl;

		train_all(epochs);

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

    auto train_csv_path = "sample_data/mnist_train_small.csv";
    int epoch_count, sample_size;
    double validation_ratio;


	cout << "Enter number of Epochs\n" << flush;
	cin >> epoch_count;

	// cout << "Enter sample size for gradient descent\n" << flush;
	// int sample_size; cin >> sample_size;

	cout << "Enter fraction of training data to be used for validation\n" << flush;
	cin >> validation_ratio;

	Optimizers::RMSProp::rate = 0.001;
	Optimizers::RMSProp::eps = 1e-5;
	Optimizers::RMSProp::decay = 0.9;

    model<Optimizers::RMSProp, LossFunctions::CrossEntropy, 10> m;
    m.train(train_csv_path, epoch_count, sample_size, validation_ratio);


}