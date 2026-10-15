#ifndef OPTIMIZER_MINDTHESTEP_H_
#define OPTIMIZER_MINDTHESTEP_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include "../Optimizer.h"

namespace MiniDNN {

// Eq. (17), K = 0: positive rates proportional to tau! / workers^tau.
// This differs from the paper's K = 1 experiment. See DAT300_Relevant_Notes.md.
class MindTheStep : public Optimizer {
    static constexpr int max_tau = 150;
    static constexpr unsigned refresh_interval = 256;
    struct Statistics {
        std::mutex mutex;
        std::array<unsigned long long, max_tau + 1> counts{};
        std::array<long double, max_tau + 1> log_weights{};
        std::array<Scalar, max_tau + 1> rates{};
        unsigned workers = 0;
        unsigned long long samples = 0;
        unsigned long long applied = 0, discarded = 0, zero = 0, capped = 0;
        long double rate_sum = 0;
    };
    // Worker clones share observations; their current step sizes remain local.
    std::shared_ptr<Statistics> stats = std::make_shared<Statistics>();
    Scalar step_size;

    void normalize() {
        long double smallest = std::numeric_limits<long double>::infinity();
        long double largest = -smallest;
        for (int i = 0; i <= max_tau; ++i) {
            if (!stats->counts[i]) continue;
            smallest = std::min(smallest, stats->log_weights[i]);
            largest = std::max(largest, stats->log_weights[i]);
        }
        // Solve E[min(exp(log_scale + log_weight), 5)] = 1.
        // These bounds bracket the solution for every observed distribution.
        long double low = -largest, high = -smallest;
        const long double log_cap = std::log(5.L);
        for (int iteration = 0; iteration < 80; ++iteration) {
            const long double middle = (low + high) / 2;
            long double sum = 0;
            for (int i = 0; i <= max_tau; ++i) {
                if (stats->counts[i])
                    sum += stats->counts[i] * std::exp(std::min(
                        middle + stats->log_weights[i], log_cap));
            }
            if (sum < stats->samples) low = middle;
            else high = middle;
        }
        const long double log_scale = (low + high) / 2;
        const long double log_base = std::log(static_cast<long double>(m_lrate));
        const long double log_floor = std::log(
            static_cast<long double>(std::numeric_limits<Scalar>::min()));
        for (int i = 0; i <= max_tau; ++i) {
            // Avoid overflow before capping and underflow to an exact zero.
            const long double log_ratio = log_scale + stats->log_weights[i];
            stats->rates[i] = log_ratio >= log_cap ? 5 * m_lrate :
                std::min(5 * m_lrate, static_cast<Scalar>(std::exp(
                    std::max(log_base + log_ratio, log_floor))));
        }
    }

public:
    struct Diagnostics {
        unsigned workers;
        unsigned long long histogram_samples, applied, discarded, zero, capped;
        Scalar mean_step_size, normalized_mean_step_size;
        std::array<unsigned long long, max_tau + 1> histogram;
    };

    explicit MindTheStep(Scalar learning_rate = 0.01) : step_size(learning_rate) {
        if (!std::isfinite(learning_rate) || learning_rate <= 0 ||
            learning_rate > std::numeric_limits<Scalar>::max() / 5)
            throw std::invalid_argument("MindTheStep requires a positive finite learning rate with a finite 5x cap");
        m_lrate = learning_rate;
    }

    MindTheStep *clone() override { return new MindTheStep(*this); }

    // Called once per gradient, before updating any parameters.
    bool prepare_step(long tau, unsigned workers) {
        std::lock_guard<std::mutex> lock(stats->mutex);
        workers = std::max(1u, workers);
        if (stats->workers != workers) {
            stats->workers = workers;
            stats->counts.fill(0);
            stats->samples = 0;
            stats->rates.fill(m_lrate); // Base-rate warmup until 256 observations.
            const long double log_workers = std::log(static_cast<long double>(workers));
            for (int i = 0; i <= max_tau; ++i)
                stats->log_weights[i] = std::lgamma(i + 1.L) - i * log_workers;
        }
        if (tau < 0 || tau > max_tau) {
            ++stats->discarded;
            step_size = 0;
            return false;
        }
        ++stats->counts[tau];
        ++stats->samples;
        if (stats->samples % refresh_interval == 0) normalize();
        step_size = stats->rates[tau];
        ++stats->applied;
        stats->rate_sum += step_size;
        if (step_size == 0) ++stats->zero;
        if (step_size >= 5 * m_lrate) ++stats->capped;
        return true;
    }

    Diagnostics diagnostics() const {
        std::lock_guard<std::mutex> lock(stats->mutex);
        long double weighted_sum = 0;
        for (int i = 0; i <= max_tau; ++i)
            weighted_sum += stats->counts[i] * stats->rates[i];
        return {stats->workers, stats->samples, stats->applied, stats->discarded,
            stats->zero, stats->capped,
            stats->applied ? static_cast<Scalar>(stats->rate_sum / stats->applied) : 0,
            stats->samples ? static_cast<Scalar>(weighted_sum / stats->samples) : 0,
            stats->counts};
    }

    void initialize_state(int) override {}
    void initialize_layer_state(int, int, int) override {}
    void update(ConstAlignedMapVec &gradient, AlignedMapVec &parameters) override {
        parameters.noalias() -= step_size * gradient;
    }
    Scalar get_w_delta(int, int, Scalar gradient) override { return -step_size * gradient; }
    Scalar get_b_delta(int, int, Scalar gradient) override { return -step_size * gradient; }
};

} // namespace MiniDNN

#endif
