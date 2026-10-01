#ifndef COMPONENT_MODEL_INTERFACES_HPP
#define COMPONENT_MODEL_INTERFACES_HPP

#include "NetworkTopology.h"
#include "modular_components.hpp"

namespace MiniDNN {

/* This is a specialisation of a model interface which creates a CNN network
 * and a momentum SGD optimizer along with it to apply updates. */
class StandardModelInterface : public ModelInterface {
protected:
    std::shared_ptr<SGDM> optimizer;
public:
    StandardModelInterface(SystemExecutor &exec, NetworkTopology ntw, double lrate, double mu, int seed, int num_threads);

    std::shared_ptr<Optimizer> get_optimizer() { return optimizer; }
};

}

#endif /* COMPONENT_MODEL_INTERFACES_HPP */
