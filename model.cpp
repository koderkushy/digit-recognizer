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
#include "include/Optimizers/Optimizers.h"
#include "include/Activations/ReLU.h"


template<
	class NeuralNet
>
struct MnistDigitRecogniser {
	NeuralNet nn {};

	std::mt19937 rng;

	MnistDigitRecogniser (): rng(std::chrono::high_resolution_clock::now().time_since_epoch().count())
	{

	}

	using image28 = nn::util::image<28, 1>;

	auto save () { nn.save("data/model.txt"); }
	// auto predict (const image28& x) { return nn.predict(x); }
	// auto recurse (const image28& x, int label) { return nn.recurse(x, label); }
	// auto evaluate (const image28& x, int label) { return nn.evaluate(x, label); }
	// auto optimize () { nn.optimize(); }

	auto train (const std::vector<std::pair<image28, int>>& train_set, const int batch)
	{

		for (auto i = begin(train_set); i + batch < end(train_set); i += batch) {
			// auto start = std::chrono::high_resolution_clock::now();

			// float batch_loss { };
			// std::mutex mut { };

			std::for_each (std::execution::par, i, i + batch, [&](const auto& data) {
				const auto& [img, label] = data;
				const auto& [gradient, loss] = nn.recurse(img, label);
				// std::lock_guard<std::mutex> lock(mut);
				// batch_loss += loss;
			});

			nn.optimize();
			// auto stop = std::chrono::high_resolution_clock::now();
			// std::cout << "Avg Time = " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (1000.0 * batch) << "ms" << std::endl;
			// std::cout << "Batch loss avg = " << batch_loss / batch << std::endl;
		}
	}

	auto validate (const std::vector<std::pair<image28, int>>& val_set) const
	{
		float val_loss { };
		std::mutex mut { };

		std::for_each (std::execution::par, begin(val_set), end(val_set), [&](const auto& data) {
			const auto& [img, label] = data;
			auto loss { nn.evaluate(img, label) };

			std::lock_guard<std::mutex> lock(mut);
			val_loss += loss;
		});

		return val_loss;
	}

	auto test (const std::vector<std::pair<image28, int>>& test_set) const
	{
		int count { };
		std::mutex mut { };

		std::for_each(std::execution::par, begin(test_set), end(test_set), [&](const auto& data) {
			const auto& [img, label] = data;
			auto predicted_class = nn.predict(img);

			if (label == predicted_class) {
				std::lock_guard<std::mutex> lock(mut);
				count++;
			}
		});

		return count * 100.0 / test_set.size();
	}
};

// template<
// 	class NeuralNet
// >
// struct model {

// 	NeuralNet nn { };

// 	std::mt19937 rng;

// 	using image28 = nn::util::image<28, 1>;

// 	std::vector<std::pair<image28, int>>
// 		train_set,
// 		val_set,
// 		test_set
// 	;

// 	model (): rng(std::chrono::high_resolution_clock::now().time_since_epoch().count()) {
// 		std::cout << "Initialised all layers\n" << std::flush;
// 	}

// 	auto save (const std::string path) {
// 		std::ofstream out(path);
// 		out.close();
// 		nn.save(path);
// 	}

// 	auto train_all (const int epochs, const std::array<float, 3> dropout_ratios = {0.5, 0.5, 0.5}) {
		
// 		float best_val_loss { std::numeric_limits<float>::max() };
// 		constexpr int jump = 16;
// 		static_assert(jump % 4 == 0);

// 		std::ofstream val_log ("data/val_loss.txt");

// 		for (int i = 0; i < epochs; i++) {
// 			std::cout << "Epoch: " << i + 1 << std::endl;

// 			std::shuffle(begin(train_set), end(train_set), rng);

// 			// float training_loss { };

// 			for (auto i = begin(train_set); i + jump < end(train_set); i += jump) {
// 				auto start = std::chrono::high_resolution_clock::now();

// 				// float sample_loss { };
// 				// std::mutex loss_mutex;

// 				std::for_each (std::execution::par, i, i + jump, [&](const auto& data) {
// 					const auto& [img, label] = data;
// 					const auto& [gradient, loss] = nn.recurse(img, label);

// 					// std::lock_guard<std::mutex> lock(loss_mutex);
// 					// sample_loss += loss;
// 				});

// 				nn.optimize();
// 				// std::cout << std::fixed << std::setprecision(3);
// 				auto stop = std::chrono::high_resolution_clock::now();
// 				std::cout << "Avg Time = " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (1000.0 * jump) << "ms\t" << std::endl;
// 				// std::cout << "Avg Loss = " << sample_loss / jump << std::endl;

// 				// training_loss += sample_loss;
// 			}

// 			std::cout << "Validating...\n" << std::flush;

// 			float val_loss { };
// 			std::mutex loss_mutex;

// 			std::for_each (std::execution::par, begin(val_set), end(val_set), [&](const auto& data) {
// 				const auto& [img, label] = data;
// 				auto loss { nn.evaluate(img, label) };
// 				std::lock_guard<std::mutex> lock(loss_mutex);
// 				val_loss += loss;
// 			});

// 			val_loss /= val_set.size();
// 			std::cout << "Validation loss = " << val_loss << '\n';
// 			val_log << val_loss << ' ' << std::flush;

// 			if (best_val_loss > val_loss)
// 				std::cout << "Previous best = " << best_val_loss << '\n',
// 				std::cout << "Saving model...\n",
// 				best_val_loss = val_loss,
// 				save("data/model.txt");
// 		}
// 	}

// 	auto train (const std::string train_csv_path, int epochs, float validation_ratio) {
// 		train_set = nn::util::load_labeled_mnist(train_csv_path);

// 		shuffle(train_set.begin(), train_set.end(), rng);
// 		auto j = train_set.end() - train_set.size() * validation_ratio;
		
// 		val_set = std::vector(j, train_set.end());
// 		train_set.erase(j, train_set.end());

// 		std::cout << "Training set has " << train_set.size() << " images\n"
// 			<< "Validation set has " << val_set.size() << " images\n" << std::endl;

// 		train_all(epochs);

// 	}

// 	auto test (const std::vector<std::pair<nn::util::image<28, 1>, int>>& test_set) {
// 		int correct_count = 0;
// 		std::mutex count_mutex;

// 		std::for_each (begin(test_set), end(test_set), [&](const auto& data) {
// 			const auto& [img, label] = data;
// 			int prediction = predict(img);
// 			if (prediction == label) {
// 				std::lock_guard<std::mutex> lock(count_mutex);
// 				correct_count += 1;
// 			}
// 		});

// 		std::cout << "Accuracy: " << correct_count * 100.0 / test_set.size() << std::endl;
// 	}

	

// 	int predict (const nn::util::filter<28>& x) {
// 		return nn.predict(x);
// 	}

	
// };

class RmsPropParams {
public:
	static constexpr float decay = 0.9;
	static constexpr float rate = 0.001;
	static constexpr float eps = 1e-7;
};



int main(){

  //   auto train_csv_path = "sample_data/mnist_train_small.csv";
  //   int epoch_count = 50;
  //   float validation_ratio = 0.2;


  //   model<
  //   	nn::Convolutional<5, 1, 32, 1, 1, nn::Optimizers::RmsProp<RmsPropParams>,
		// nn::Convolutional<5, 32, 32, 1, 1, nn::Optimizers::RmsProp<RmsPropParams>, nn::ReLU<nn::MaxPool<2, 0, 1, nn::DropOut<50,
		// nn::Convolutional<3, 32, 64, 1, 1, nn::Optimizers::RmsProp<RmsPropParams>,
		// nn::Convolutional<3, 64, 64, 1, 1, nn::Optimizers::RmsProp<RmsPropParams>, nn::ReLU<nn::MaxPool<2, 0, 2, nn::DropOut<50,
		// nn::FullyConnected<11 * 11 * 64, 2048, nn::Optimizers::RmsProp<RmsPropParams>, nn::ReLU<nn::DropOut<50,
		// nn::FullyConnected<2048, 10, nn::Optimizers::RmsProp<RmsPropParams>, nn::OutputLayer<10, LossFunctions::CrossEntropy>>>>>>>>>>>>>>>
  //   > m;
  //   m.train(train_csv_path, epoch_count, validation_ratio);

		using Optimizer = nn::Optimizers::RmsProp<RmsPropParams>;

	 //    MnistDigitRecogniser<
	 //    	nn::Convolutional<5, 1, 32, 0, 1, Optimizer,
		// 	nn::Convolutional<5, 32, 32, 0, 1, Optimizer, nn::ReLU<nn::MaxPool<2, 0, 1, nn::DropOut<50,
		// 	nn::Convolutional<3, 32, 64, 0, 1, Optimizer,
		// 	nn::Convolutional<3, 64, 64, 0, 1, Optimizer, nn::ReLU<nn::MaxPool<2, 0, 2, nn::DropOut<50,
		// 	nn::FullyConnected<7 * 7 * 64, 2048, Optimizer, nn::ReLU<nn::DropOut<50,
		// 	nn::FullyConnected<2048, 10, Optimizer, nn::OutputLayer<10, LossFunctions::CrossEntropy>>>>>>>>>>>>>>>
		// > model{ };

		MnistDigitRecogniser<
			nn::Convolutional<3, 1, 32, 0, Optimizer, nn::ReLU<
			nn::Convolutional<3, 32, 32, 0, Optimizer, nn::ReLU<nn::MaxPool<3, 0, 2, nn::DropOut<50,
			nn::Convolutional<3, 32, 32, 0, Optimizer, nn::ReLU<
			nn::Convolutional<3, 32, 64, 0, Optimizer, nn::ReLU<nn::MaxPool<2, 0, 1, nn::DropOut<50,
			nn::Convolutional<3, 64, 64, 1, Optimizer, nn::ReLU<
			nn::Convolutional<3, 64, 128, 1, Optimizer, nn::ReLU<
			nn::Convolutional<3, 128, 128, 1, Optimizer, nn::ReLU<
			nn::Convolutional<3, 128, 128, 1, Optimizer, nn::ReLU<nn::MaxPool<2, 0, 1, nn::DropOut<50,
			nn::FullyConnected<5 * 5 * 128, 2048, Optimizer, nn::ReLU<
			nn::FullyConnected<2048, 10, Optimizer, nn::OutputLayer<10, LossFunctions::CrossEntropy>>>>>>>>>>>>>>>>>>>>>>>>>>
		> model{ };

		std::mt19937 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());

		auto train_set = nn::util::load_labeled_mnist("sample_data/mnist_train_small.csv");
		std::shuffle(begin(train_set), end(train_set), rng);

		auto test_set = std::vector(end(train_set) - 0.2 * train_set.size(), end(train_set));
		train_set.erase(end(train_set) - 0.2 * train_set.size(), end(train_set));

		// auto val_set = std::vector(end(train_set) - 0.2 * train_set.size(), end(train_set));
		// train_set.erase(end(train_set) - 0.2 * train_set.size(), end(train_set));


		std::cout << "Training set has " << train_set.size() << " images\n";
		// std::cout << "Validation set has " << val_set.size() << " images\n";
		std::cout << "Test set has " << test_set.size() << " images\n" << std::endl;


		float least_val_loss = std::numeric_limits<float>::max();

		for (int epoch = 0; epoch < 5; epoch++) {
			std::cout << "Epoch " << epoch + 1 << std::endl;

			model.train(train_set, 32);

			// if (model.validate(val_set) < least_val_loss)
			// 	model.save();

			std::cout << "Accuracy " << model.test(test_set) << " %" << std::endl;
		}




}
