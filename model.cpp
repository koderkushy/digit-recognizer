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
	auto predict (const image28& x) { return nn.predict(x); }
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


class RmsPropParams {
public:
	static constexpr float decay = 0.9;
	static constexpr float rate = 0.001;
	static constexpr float eps = 1e-7;
};



int main(){

  
	using Optimizer = nn::Optimizers::RmsProp<RmsPropParams>;


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

	auto train_set = nn::util::load_labeled_mnist("sample_data/train.csv");
	std::shuffle(begin(train_set), end(train_set), rng);

	// auto test_set = std::vector(end(train_set) - 0.2 * train_set.size(), end(train_set));
	// train_set.erase(end(train_set) - 0.2 * train_set.size(), end(train_set));

	auto val_set = std::vector(end(train_set) - 0.2 * train_set.size(), end(train_set));
	train_set.erase(end(train_set) - 0.2 * train_set.size(), end(train_set));


	std::cout << "Training set has " << train_set.size() << " images\n";
	std::cout << "Validation set has " << val_set.size() << " images\n";
	// std::cout << "Test set has " << test_set.size() << " images\n" << std::endl;


	float least_val_loss = std::numeric_limits<float>::max();

	for (int epoch = 0; epoch < 5; epoch++) {
		std::cout << "Epoch " << epoch + 1 << std::endl;

		std::shuffle(begin(train_set), end(train_set), rng);

		model.train(train_set, 64);

		// if (model.validate(val_set) < least_val_loss)
		// 	model.save();

		std::cout << "Accuracy " << model.test(val_set) << " %" << std::endl;
	}

	auto test_set = nn::util::load_unlabeled_mnist("sample_data/test.csv");

	std::cout << "Test set has " << test_set.size() << " images" << std::endl;

	std::vector predictions(test_set.size(), 0);
	std::vector iota(test_set.size(), 0);

	std::iota(begin(iota), end(iota), 0);

	std::for_each(std::execution::par, begin(iota), end(iota), [&](const int i) {
		predictions[i] = model.predict(test_set[i]);
	});

	std::ofstream out("sample_data/predictions.csv");

	out << "ImageId" << ',' << "Label" << '\n';

	for (int i = 0; i < predictions.size(); i++) {
		out << i + 1 << ',' << predictions[i] << '\n';
	}

}