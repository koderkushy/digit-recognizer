#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2")

#include "bits/stdc++.h"
#include <execution>

using namespace std;

template <typename A, typename B> ostream& operator<< (ostream &cout, pair<A, B> const &p) { return cout << "(" << p.first << ", " << p.second << ")"; }
template <typename A, typename B> istream& operator>> (istream& cin, pair<A, B> &p) {cin >> p.first; return cin >> p.second;}
template <typename A> ostream& operator<< (ostream &cout, vector<A> const &v) {cout << "["; for(int i = 0; i < v.size(); i++) {if (i) cout << ", "; cout << v[i];} return cout << "]";}
template <typename A> istream& operator>> (istream& cin, vector<A> &x){for(int i = 0; i < x.size()-1; i++) cin >> x[i]; return cin >> x[x.size()-1];}
template <typename A, typename B> A amax (A &a, B b){ if (b > a) a = b ; return a; }
template <typename A, typename B> A amin (A &a, B b){ if (b < a) a = b ; return a; }

#include "matmul.h"

using namespace operations;

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

		if constexpr (std::min({N, M, K}) > 50)
			FastMatMul::fast_mat_mul(A, B, C);
		else
			for (int i = 0; i < M; i++)
					for (int k = 0; k < N; k++)
#pragma GCC ivdep
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
#include "FullyConnectedLayer.h"
#include "LossFunctions.h"
#include "Optimizers.h"
#include "ReLU.h"

template<
	int classes,
	class Loss
>
struct OutPutLayer {

	template<uint64_t N, uint64_t channels>
	auto recurse (const image<N, channels>& Y_img, int label) {
		static_assert(N == 1 and channels == classes);
		auto Y { array_converted(Y_img) };
		return pair(imagify<1, 10, 10>(Loss::gradient(Y, label)), Loss::loss(Y, label));
	}
	
	template<uint64_t N, uint64_t channels>
	auto evaluate (const image<N, channels>& X, const int label) {
		static_assert(N == 1 and channels == classes);
		return Loss::loss(array_converted(X), label);
	}

	void optimize () {}
	void save (const string path) {}
};

template<
	class Optimizer,
	class Loss
>
struct model {

	ConvolutionalLayer<Optimizer, 1, 32, 5, 1,
		ConvolutionalLayer<Optimizer, 32, 32, 5, 1,
			ReLU<
				MaxPool<2, 0, 1,
					DropOut<50,
						ConvolutionalLayer<Optimizer, 32, 64, 3, 1,
							ConvolutionalLayer<Optimizer, 64, 64, 3, 1,
								ReLU<
									MaxPool<2, 0, 2,
										DropOut<50,
											FullyConnectedLayer<Optimizer, 11, 64, 32, 2,
												ReLU<
													DropOut<50,
														FullyConnectedLayer<Optimizer, 32, 2, 1, 10,
															OutPutLayer<10, Loss>>>>>>>>>>>>>>>
																nn { };

	mt19937 rng;

	using image28 = image<28, 1>;

	vector<pair<image28, int>> train_set, validation_set, test_set;

	model (): rng(chrono::high_resolution_clock::now().time_since_epoch().count()) {
		cout << "Initialised all layers\n" << flush;
		save("initial.txt");
	}

	auto save (const string path) {
		ofstream out(path);
		out.close();
		// nn.save(path);
	}

	auto train_all (const int epochs, const array<double, 3> dropout_ratios = {0.5, 0.5, 0.5}) {
		
		double best_validation_loss { std::numeric_limits<double>::max() };
		constexpr int jump = 8;
		static_assert(jump % 4 == 0);

		cout << "jump " << jump << '\n';

		for (int i = 0; i < epochs; i++) {
			cout << "Epoch: " << i + 1 << endl;

			shuffle(begin(train_set), end(train_set), rng);

			double training_loss { };

			for (auto i = begin(train_set); i + jump < end(train_set); i += jump) {
				auto start = chrono::high_resolution_clock::now();

				double sample_loss { };
				std::mutex loss_mutex;

				for_each (execution::par, i, i + jump, [&](const auto& data) {
					const auto& [image, label] = data;
					const auto& [gradient, loss] = nn.recurse(image, label);

					lock_guard<mutex> lock(loss_mutex);
					sample_loss += loss;
				});

				nn.optimize();
				cout << fixed << setprecision(3);
				auto stop = chrono::high_resolution_clock::now();
				cout << "Avg Time = " << chrono::duration_cast<chrono::microseconds>(stop - start).count() / (1000.0 * jump) << "ms\t";
				cout << "Avg Loss = " << sample_loss / jump << endl;

				training_loss += sample_loss;
			}

			cout << "Validating...\n" << flush;

			double validation_loss { };
			std::mutex loss_mutex;

			for_each(execution::par, begin(validation_set), end(validation_set), [&](const auto& data) {
				const auto& [img, label] = data;
				auto loss { nn.evaluate(img, label) };
				lock_guard<mutex> lock(loss_mutex);
				validation_loss += loss;
			});

			validation_loss /= validation_set.size();
			cout << "Validation loss = " << validation_loss << '\n';

			if (best_validation_loss > validation_loss)
				cout << "Previous best = " << best_validation_loss << '\n',
				cout << "Saving model...\n",
				best_validation_loss = validation_loss,
				save("model_parameters/model.txt");
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

	}

	auto load_data (const string csv_path) {
		if (freopen(csv_path.c_str(), "r", stdin) == NULL)
			cout << "Couldn't open file.\n", exit(0);

		constexpr int N = 28;

		vector<pair<image<N, 1>, int>> set{};
		string s, word;

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
};

int main(){

    auto train_csv_path = "sample_data/mnist_train_small.csv";
    int epoch_count = 10000, sample_size = 120;
    double validation_ratio = 0.2;


	// cout << "Enter number of Epochs\n" << flush;
	// cin >> epoch_count;

	// cout << "Enter sample size for gradient descent\n" << flush;
	// int sample_size; cin >> sample_size;

	// cout << "Enter fraction of training data to be used for validation\n" << flush;
	// cin >> validation_ratio;

	Optimizers::RMSProp::rate = 0.001;
	Optimizers::RMSProp::eps = 1e-5;
	Optimizers::RMSProp::decay = 0.9;

    model<Optimizers::RMSProp, LossFunctions::CrossEntropy> m;
    m.train(train_csv_path, epoch_count, sample_size, validation_ratio);


}

/*
*/