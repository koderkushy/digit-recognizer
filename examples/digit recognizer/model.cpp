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

#include "../../include/Layers/ConvolutionalLayer.h"
#include "../../include/Layers/FullyConnectedLayer.h"
#include "../../include/Layers/DropOut.h"
#include "../../include/Layers/MaxPooling.h"
#include "../../include/Layers/OutputLayer.h"
#include "../../include/Activation Functions/ReLU.h"
#include "../../include/Loss Functions/LossFunctions.h"
#include "../../include/Optimizers/Optimizers.h"


template<
	class Optimizer,
	class Loss
>
struct model {

	nn::ConvolutionalLayer<Optimizer, 1, 32, 5, 1,
		nn::ConvolutionalLayer<Optimizer, 32, 32, 5, 1,
			nn::ReLU<
				nn::MaxPool<2, 0, 1,
					nn::DropOut<50,
						nn::ConvolutionalLayer<Optimizer, 32, 64, 3, 1,
							nn::ConvolutionalLayer<Optimizer, 64, 64, 3, 1,
								nn::ReLU<
									nn::MaxPool<2, 0, 2,
										nn::DropOut<50,
											nn::FullyConnectedLayer<Optimizer, 11, 64, 32, 2,
												nn::ReLU<
													nn::DropOut<50,
														nn::FullyConnectedLayer<Optimizer, 32, 2, 1, 10,
															nn::OutputLayer<10, Loss>>>>>>>>>>>>>>>
																nn { };

	mt19937 rng;

	vector<pair<nn::operations::image<28, 1>, int>> train_set, validation_set, test_set;

	model (): rng(chrono::high_resolution_clock::now().time_since_epoch().count()) {
		cout << "Initialised all layers\n" << flush;
		nn.save("initial");
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
					const auto& [img, label] = data;
					const auto& [gradient, loss] = nn.recurse(img, label);

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

		// train_all(epochs);

	}

	auto test (const vector<pair<nn::operations::image<28, 1>, int>>& test_set) {

	}

	auto load_data (const string csv_path) {
		if (freopen(csv_path.c_str(), "r", stdin) == NULL)
			cout << "Couldn't open file.\n", exit(0);

		constexpr int N = 28;

		vector<pair<nn::operations::image<N, 1>, int>> set{};
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

			nn::operations::image<N, 1> img{};
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

	nn::optimizers::RMSProp::rate = 0.001;
	nn::optimizers::RMSProp::eps = 1e-5;
	nn::optimizers::RMSProp::decay = 0.9;

    model<nn::optimizers::RMSProp, nn::lossfunctions::CrossEntropy> m;
    m.train(train_csv_path, epoch_count, sample_size, validation_ratio);


}

/*
*/