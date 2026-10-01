#include "ParameterContainer.h"
#include "minidnn/Component/Worker.hpp"
#include <chrono>

// #define DUMMY

namespace MiniDNN {

void SGDWorkerAsync::run() {
    /* Delay thread execution until flag is tripped */
    //this->flag->wait(true); // Nowadays, this is handled by the worker pool

    // If this starts running before the Dispatcher is ready to actually start
    // (which is likely since threads are created at initialisation), that's okay
    // as long as Dispatcher.is_finished() is not true.
    // Since try_start_step could initially block until the Dispatcher is ready,
    // this therefore doesn't have to involve busy waiting.
    while (!_exec.get_dispatcher()->is_finished()) {
        const auto [can_start, step_ind] = _exec.get_dispatcher()->try_start_step(this->_id);
        if (can_start) {
            // if (!outstanding_step) {
            //     outstanding_step = true;
            //     t0 = HRClock::now();
            // }

            const auto t1 = HRClock::now();

            // Calculate a gradient based on this batch (getting loss)
            auto global_param_ptr = _exec.get_model()->get_network()->current_param_container_ptr;
            auto *local_param = new ParameterContainer(*global_param_ptr);

            const long param_version_start = local_param->timestamp;

            #ifdef DUMMY // Simulate larger variance in time
            // auto dist = std::normal_distribution<double>(20, 0.25);
            // auto dist = std::normal_distribution<double>(20, 1);
            // auto dist = std::normal_distribution<double>(20, 2);
            auto dist = std::normal_distribution<double>(20, 5);
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(dist(_rng)));
            #else // Do actual work
            // Get batch from batch controller
            int batch_sz;
            const auto batch_id = _exec.get_batcher()->get_batch_ind(this->_id,
                std::make_unique<int>(batch_sz));
            const Matrix &b_x = _exec.get_batcher()->get_batch_data(batch_id, batch_sz);
            const Matrix &b_y = _exec.get_batcher()->get_batch_labels(batch_id, batch_sz);

            this->network->set_pointer(local_param);
            this->network->forward(b_x);
            this->network->backprop(b_x, b_y);
            #endif


            const double local_loss = this->network->get_loss();
            long step_end_ind;

            if (_exec.get_dispatcher()->finish_step(this->_id, step_ind, step_end_ind)) {
                // outstanding_step = false;

                accepted_steps++;
                
                // Apply gradient to model interface
                // TODO: This section should really be delegated to the ModelInterface
                this->network->set_pointer(global_param_ptr);
                delete local_param;

                const long param_version_end = global_param_ptr->timestamp;
                long tau = param_version_end - param_version_start;
                this->optim->tau = tau;

                this->network->update_cw(this->optim.get());

                const auto t2 = HRClock::now();

                const long x = (t2 - t1).count();

                // Give loss to monitor
                _exec.get_monitor()->update(local_loss, x, step_end_ind);

                // Record loss in local epoch bucket
                int epoch_now = step_end_ind / _exec._steps_per_epoch;
                if (epoch_now + 1 > _epoch_loss_sums.size()) {
                    _epoch_loss_sums.resize(epoch_now + 1);
                    _epoch_recorded_steps.resize(epoch_now + 1);
                }
                _epoch_loss_sums.at(epoch_now) += local_loss;
                _epoch_recorded_steps.at(epoch_now)++;

                if (tau < MAX_TAU_DIST) {
                    _tau_distr.at(tau) += 1;
                }

#ifdef MEASURE_TAU_PER_EPOCH
                if (tau < MAX_MEASURE_TAU_PER_EPOCH) {
                    if (epoch_now >= _epoch_tau_distr.size()) {
                        _epoch_tau_distr.resize(epoch_now + 1);
                    }
                    _epoch_tau_distr.at(epoch_now).at(tau) += 1;
                }
#endif

#if MEASURE_STEP_TIME
                /* If we want to print all the measured time samples afterwards, we have to store them. */

                // Append new samples, up to vector's reserved size
                if (steptime_samples.size() < N_STEP_TIME_SAMPLES) {
                    const auto t_start = _exec.start_time_hr;
                    steptime_samples.emplace_back((t1-t_start).count(), (t2-t_start).count(), this->_id, tau);
                }
#endif
            } else {
                rejected_steps++;
                delete local_param;
            }
        }
    }

#if MEASURE_STEP_TIME
    _exec.submit_steptimes(steptime_samples);
#endif

    _exec.submit_tau_dist(_tau_distr);
    _exec.submit_acceptance_rate(accepted_steps, rejected_steps);
    _exec.submit_epoch_tau_dist(_epoch_tau_distr);
    _exec.submit_thread_epochs(_epoch_loss_sums, _epoch_recorded_steps);
}

void SGDWorkerSynchronous::run() {
    const auto [can_start, step_ind] = _exec.get_dispatcher()->try_start_step(this->_id);
    if (!can_start) return;

    // Perform just 1 (one) SGD step

    const auto t1 = HRClock::now();
    int batch_sz;
    const auto batch_id = _exec.get_batcher()->get_batch_ind(this->_id,
                std::make_unique<int>(batch_sz));
    const Matrix &b_x = _exec.get_batcher()->get_batch_data(batch_id, batch_sz);
    const Matrix &b_y = _exec.get_batcher()->get_batch_labels(batch_id, batch_sz);

    // Calculate a gradient based on this batch (getting loss)
    auto global_param_ptr = _exec.get_model()->get_network()->current_param_container_ptr;
    auto *local_param = new ParameterContainer(*global_param_ptr);

    const long param_version_start = global_param_ptr->timestamp;

    this->network->set_pointer(local_param);
    this->network->forward(b_x);
    this->network->backprop(b_x, b_y);

    const double local_loss = this->network->get_loss();

    // Apply gradient to model interface
    // TODO: This section should really be delegated to the ModelInterface
    this->network->set_pointer(global_param_ptr);
    delete local_param;

    const long param_version_end = global_param_ptr->timestamp;
    long tau = param_version_end - param_version_start;

    this->network->update_cw(this->optim.get());

    long end = step_ind;
    const long finished_step = _exec.get_dispatcher()->finish_step(this->_id, step_ind, end);

    const auto t2 = HRClock::now();
    const long x = (t2 - t1).count();

    // Give loss to monitor
    _exec.get_monitor()->update(local_loss, x, finished_step);

    if (tau < MAX_TAU_DIST) {
        // ... TODO: Make an array for tau distribution, and update it from here
    }
    
#if MEASURE_STEP_TIME
    /* If we want to print all the measured time samples afterwards, we have to store them. */

    // Append new samples, up to vector's reserved size
    if (steptime_samples.size() < N_STEP_TIME_SAMPLES) {
        const auto t_start = _exec.start_time_hr;
        steptime_samples.emplace_back((t1-t_start).count(), (t2-t_start).count(), this->_id, tau);
    }
#endif
}

}
