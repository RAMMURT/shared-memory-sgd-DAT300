#ifndef COMPONENT_MODEL_INTERFACES_HPP
#define COMPONENT_MODEL_INTERFACES_HPP

#include "NetworkTopology.h"
#include "modular_components.hpp"
#include "Optimizer/MindTheStep.h"

namespace MiniDNN {

/* This is a specialisation of a model interface which creates a CNN network
 * and a MindTheStep optimizer along with it to apply updates. */
class StandardModelInterface : public ModelInterface {
protected:
    std::shared_ptr<MindTheStep> optimizer;
public:
    StandardModelInterface(SystemExecutor &exec, NetworkTopology ntw, double lrate, double mu, int seed);

    std::shared_ptr<Optimizer> get_optimizer() { return optimizer; }
};

}

#endif /* COMPONENT_MODEL_INTERFACES_HPP */
