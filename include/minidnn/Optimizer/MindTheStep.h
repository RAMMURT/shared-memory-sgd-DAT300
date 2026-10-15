#ifndef OPTIMIZER_MINDTHESTEP_H_
#define OPTIMIZER_MINDTHESTEP_H_

#include <algorithm>
#include <cmath>
#include "../Optimizer.h"

namespace MiniDNN {

// MindTheStep-APSGD, Eq. (17): Poisson delays, lambda = workers, K = 1.
// Uses the paper's 5*alpha cap and tau > 150 cutoff (Section VI).
// The empirical mean-step normalization of Eq. (26) is not included.
class MindTheStep : public Optimizer {
    Scalar step_size = 0;

public:
    explicit MindTheStep(Scalar learning_rate = 0.01) {
        m_lrate = learning_rate;
        step_size = learning_rate;
    }

    MindTheStep *clone() override { return new MindTheStep(*this); }

    // Call once per gradient, before updating any of its parameters.
    bool prepare_step(long tau, unsigned workers) {
        if (tau < 0 || tau > 150) return false;
        const long double lambda = std::max(1u, workers);
        const long double poisson_zero = std::exp(-lambda);
        long double rate = m_lrate;
        // Equivalent to (alpha - PoissonCDF(tau-1)) * tau! / lambda^tau.
        // At tau = 0 the CDF is zero. Negative rates from Eq. (17) are retained.
        for (long i = 1; i <= tau; ++i)
            rate = (rate - poisson_zero) * i / lambda;
        step_size = static_cast<Scalar>(std::min(rate, 5.L * m_lrate));
        return true;
    }

    void initialize_state(int) override {}
    void initialize_layer_state(int, int, int) override {}

    void update(ConstAlignedMapVec &gradient, AlignedMapVec &parameters) override {
        parameters.noalias() -= step_size * gradient;
    }

    Scalar get_w_delta(int, int, Scalar gradient) override {
        return -step_size * gradient;
    }

    Scalar get_b_delta(int, int, Scalar gradient) override {
        return -step_size * gradient;
    }
};

} // namespace MiniDNN

#endif
