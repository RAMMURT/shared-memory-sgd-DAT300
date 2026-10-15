#ifndef OPTIMIZER_MINDTHESTEP_H_
#define OPTIMIZER_MINDTHESTEP_H_

#include <algorithm>
#include <cmath>
#include <array>
#include <memory>
#include <mutex>
#include <stdexcept>
#include "../Optimizer.h"

namespace MiniDNN {

// MindTheStep-APSGD, Eq. (17): Poisson delays, lambda = workers, K = 1.
// Uses the paper's 5*alpha cap and tau > 150 cutoff (Section VI).
// Online Eq. (26) normalization, with nonnegative rates for stability.
class MindTheStep : public Optimizer {
    static constexpr int max_tau = 150;
    static constexpr unsigned refresh_interval = 256;
    struct Statistics {
        std::mutex mutex;
        std::array<unsigned long long, max_tau + 1> counts{};
        std::array<long double, max_tau + 1> rates{};
        unsigned workers = 0;
        unsigned long long samples = 0, discarded = 0;
        long double normalization = 1;
    };
    // Worker clones share observations, but each keeps its own current step.
    std::shared_ptr<Statistics> stats = std::make_shared<Statistics>();
    Scalar step_size = 0;

public:
    explicit MindTheStep(Scalar learning_rate = 0.01) {
        if (!std::isfinite(learning_rate) || learning_rate <= 0)
            throw std::invalid_argument("MindTheStep requires a positive finite learning rate");
        m_lrate = learning_rate;
        step_size = learning_rate;
    }

    MindTheStep *clone() override { return new MindTheStep(*this); }

    // Call once per gradient, before updating any of its parameters.
    bool prepare_step(long tau, unsigned workers) {
        std::lock_guard<std::mutex> lock(stats->mutex);
        workers = std::max(1u, workers);
        if (stats->workers != workers) {
            stats->workers = workers;
            stats->counts.fill(0);
            stats->samples = stats->discarded = 0;
            stats->normalization = 1;
            const long double lambda = workers;
            const long double poisson_zero = std::exp(-lambda);
            long double rate = m_lrate;
            for (int i = 0; i <= max_tau; ++i) {
                // Eq. (17), with an explicit stability policy beyond the paper:
                // suppress negative rates and cap before estimating the mean.
                stats->rates[i] = std::clamp(rate, 0.L, 5.L * m_lrate);
                rate = (rate - poisson_zero) * (i + 1) / lambda;
            }
        }
        if (tau < 0 || tau > max_tau) {
            ++stats->discarded;
            step_size = 0;
            return false;
        }
        ++stats->counts[tau];
        ++stats->samples;
        if (stats->samples % refresh_interval == 0) {
            long double sum = 0;
            for (int i = 0; i <= max_tau; ++i)
                sum += stats->counts[i] * stats->rates[i];
            // A zero mean cannot be normalized; keep factor 1 in that case.
            stats->normalization = sum > 0 ? m_lrate * stats->samples / sum : 1;
        }
        // Final safety cap takes priority over exact mean-step normalization.
        step_size = static_cast<Scalar>(std::min(
            stats->rates[tau] * stats->normalization, 5.L * m_lrate));
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
