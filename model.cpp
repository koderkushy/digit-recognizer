#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2")

#include "bits/stdc++.h"
#include <execution>

#include "include/Utilities/Utilities.h"
#include "include/Math/NnMath.h"

#include "include/Layers/Convolutional.h"
#include "include/Layers/MaxPooling.h"
#include "include/Layers/DropOut.h"
#include "include/Layers/FullyConnected.h"
#include "include/Layers/OutputLayer.h"
#include "include/Losses/CrossEntropy.h"
#include "include/Optimizers/RMSProp.h"
#include "include/Activations/ReLU.h"

template<
	class Optimizer,
	class Loss
>
struct model {

	nn::Convolutional<Optimizer, 1, 32, 5, 1,
		nn::Convolutional<Optimizer, 32, 32, 5, 1,
			nn::ReLU<
				nn::MaxPool<2, 0, 1,
					nn::DropOut<50,
						nn::Convolutional<Optimizer, 32, 64, 3, 1,
							nn::Convolutional<Optimizer, 64, 64, 3, 1,
								nn::ReLU<
									nn::MaxPool<2, 0, 2,
										nn::DropOut<50,
											nn::FullyConnected<Optimizer, 11 * 11 * 64, 2048,
												nn::ReLU<
													nn::DropOut<50,
														nn::FullyConnected<Optimizer, 2048, 10,
															nn::OutputLayer<10, Loss>>>>>>>>>>>>>>>
																nn { };

	std::mt19937 rng;

	using image28 = nn::util::image<28, 1>;

	std::vector<std::pair<image28, int>> train_set, validation_set, test_set;

	model (): rng(std::chrono::high_resolution_clock::now().time_since_epoch().count()) {
		std::cout << "Initialised all layers\n" << std::flush;
	}

	auto save (const std::string path) {
		std::ofstream out(path);
		out.close();
		nn.save(path);
	}

	auto train_all (const int epochs, const std::array<double, 3> dropout_ratios = {0.5, 0.5, 0.5}) {
		
		double best_validation_loss { std::numeric_limits<double>::max() };
		constexpr int jump = 16;
		static_assert(jump % 4 == 0);

		std::ofstream validation_log ("data/validation_loss.txt");

		for (int i = 0; i < epochs; i++) {
			std::cout << "Epoch: " << i + 1 << std::endl;

			std::shuffle(begin(train_set), end(train_set), rng);

			// double training_loss { };

			for (auto i = begin(train_set); i + jump < end(train_set); i += jump) {
				// auto start = std::chrono::high_resolution_clock::now();

				// double sample_loss { };
				// std::mutex loss_mutex;

				std::for_each (std::execution::par, i, i + jump, [&](const auto& data) {
					const auto& [img, label] = data;
					const auto& [gradient, loss] = nn.recurse(img, label);

					// std::lock_guard<std::mutex> lock(loss_mutex);
					// sample_loss += loss;
				});

				nn.optimize();
				// std::cout << std::fixed << std::setprecision(3);
				// auto stop = std::chrono::high_resolution_clock::now();
				// std::cout << "Avg Time = " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (1000.0 * jump) << "ms\t";
				// std::cout << "Avg Loss = " << sample_loss / jump << std::endl;

				// training_loss += sample_loss;
			}

			std::cout << "Validating...\n" << std::flush;

			double validation_loss { };
			std::mutex loss_mutex;

			std::for_each (std::execution::par, begin(validation_set), end(validation_set), [&](const auto& data) {
				const auto& [img, label] = data;
				auto loss { nn.evaluate(img, label) };
				std::lock_guard<std::mutex> lock(loss_mutex);
				validation_loss += loss;
			});

			validation_loss /= validation_set.size();
			std::cout << "Validation loss = " << validation_loss << '\n';
			validation_log << validation_loss << ' ' << std::flush;

			if (best_validation_loss > validation_loss)
				std::cout << "Previous best = " << best_validation_loss << '\n',
				std::cout << "Saving model...\n",
				best_validation_loss = validation_loss,
				save("data/model.txt");
		}
	}

	auto train (const std::string train_csv_path, int epochs, double validation_ratio) {
		train_set = load_data(train_csv_path);

		shuffle(train_set.begin(), train_set.end(), rng);
		auto j = train_set.end() - train_set.size() * validation_ratio;
		
		validation_set = std::vector(j, train_set.end());
		train_set.erase(j, train_set.end());

		std::cout << "Training set has " << train_set.size() << " images\n"
			<< "Validation set has " << validation_set.size() << " images\n" << std::endl;

		train_all(epochs);

	}

	auto test (const std::vector<std::pair<nn::util::image<28, 1>, int>>& test_set) {

	}

	auto load_data (const std::string csv_path) {
		if (freopen(csv_path.c_str(), "r", stdin) == NULL)
			std::cout << "Couldn't open file.\n", exit(0);

		constexpr int N = 28;

		std::vector<std::pair<nn::util::image<N, 1>, int>> set{};
		std::string s, word;

		while (std::cin >> s) {
			std::vector<int> s_split{};
			s_split.reserve(N * N + 1);
			std::stringstream ss(s);

			while (!ss.eof())
				getline(ss, word, ','),
				s_split.push_back(stoi(word));

			if (s_split.size() != N * N + 1)
				std::cout << "Incorrect file format.\n", exit(0);

			nn::util::image<N, 1> img{};
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					img[0][i][j] = s_split[i * N + j + 1] / 256.0;

			auto &label = s_split[0];

			set.push_back(std::pair(img, label));
		}

		return std::move(set);
	}
};

int main(){

    auto train_csv_path = "sample_data/mnist_train_small.csv";
    int epoch_count = 50;
    double validation_ratio = 0.2;

	Optimizers::RMSProp::rate = 0.001;
	Optimizers::RMSProp::eps = 1e-5;
	Optimizers::RMSProp::decay = 0.9;

    model<Optimizers::RMSProp, LossFunctions::CrossEntropy> m;
    m.train(train_csv_path, epoch_count, validation_ratio);


}
