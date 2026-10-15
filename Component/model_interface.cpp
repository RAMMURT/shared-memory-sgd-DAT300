#include "NetworkTopology.h"
#include "Output/MultiClassEntropy.h"
#include "ParameterContainer.h"
#include "modular_components.hpp"
#include "Component/ModelInterfaces.hpp"

namespace MiniDNN {

StandardModelInterface::StandardModelInterface(SystemExecutor &exec, NetworkTopology ntw,  double lrate, double mu, int seed)
    : ModelInterface(exec, ntw) {
    // const std::string &dataset_name = exec.get_batcher()->get_dataset_name();
    //
    // // Construct neural network
    // if (dataset_name == "CIFAR10" || dataset_name == "CIFAR100") {
    //     network.add_layer(new Convolutional<ReLU>(32, 32, 3, 6, 5, 5));
    //     network.add_layer(new MaxPooling<ReLU>(28, 28, 6, 2, 2));
    //
    //     network.add_layer(new Convolutional<ReLU>(14, 14, 6, 16, 5, 5));
    //     network.add_layer(new MaxPooling<ReLU>(10, 10, 16, 2, 2));
    //
    //     network.add_layer(new FullyConnected<ReLU>(5 * 5 * 16, 120));
    // } else if (dataset_name == "MNIST" || dataset_name == "FASHION-MNIST") {
    //
    //     network.add_layer(new Convolutional<ReLU>(28, 28, 1, 6, 5, 5));
    //     network.add_layer(new MaxPooling<ReLU>(24, 24, 6, 2, 2));
    //
    //     network.add_layer(new Convolutional<ReLU>(12, 12, 6, 16, 5, 5<));
    //     network.add_layer(new MaxPooling<ReLU>(8, 8, 16, 2, 2));
    //
    //     network.add_layer(new FullyConnected<ReLU>(4 * 4 * 16, 120));
    // }
    //
    // network.add_layer(new FullyConnected<Softmax>(120, exec.get_batcher()->get_y_dimension()));
    //
    // network.set_output(new MultiClassEntropy());
    // network.init(0, 0.01, seed);

    // Create optimizer
    // Hard-coded for the mind branch; mu is unused (no explicit momentum).
    this->optimizer = std::make_shared<MindTheStep>(lrate);
}

} // namespace MiniDNN
