#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <cmath>
#include <Eigen/Core>
#include "Config.h"

namespace MiniDNN {


///
/// \defgroup Optimizers Optimization Algorithms
///

///
/// \ingroup Optimizers
///
/// The interface of optimization algorithms
///
    class Optimizer {

    protected:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
        typedef Vector::AlignedMapType AlignedMapVec;

        double alpha() const {
            return 1.;

            // Note: this is ugly code. I know. Consider it a proof of concept :P
            // set up constants
            double e = 2.718281828;
            double c = 1.5;
            double ln_2c = 1.09861; // assume c = 1.5
            double k = 16; // k = num_threads
            double n = ln_2c / k;

            // f(x) = 0.5 + c * e ^(((ln 3) / k) * x)
            // Call like f(tau)
            // This formula is set up such that f(0) = 2, f(k) = 1, f(inf) = 0.5
            return 0.5 + 1.5 * std::pow(e, -tau * n);

            // return 1. / std::max(std::sqrt(tau), 1.);
        }

    public:
        Scalar m_lrate;

        float step_scale_factor = 1.0;

        long tau = 0;

        Optimizer() {}

        virtual Optimizer *clone() = 0;

        virtual ~Optimizer() {}

        ///
        /// Reset the optimizer to clear all historical information
        ///
        virtual void reset() {};

        ///
        /// Update the parameter vector using its gradient
        ///
        /// It is assumed that the memory addresses of `dvec` and `vec` do not
        /// change during the training process. This is used to implement optimization
        /// algorithms that have "memories". See the AdaGrad algorithm for an example.
        ///
        /// \param dvec The gradient of the parameter. Read-only
        /// \param vec  On entering, the current parameter vector. On exit, the
        ///             updated parameters.
        ///
        virtual void update(ConstAlignedMapVec &dvec, AlignedMapVec &vec) = 0;

        virtual void initialize_state(int num_layers) = 0;

        virtual void initialize_layer_state(int layer_n, int m_dw_size, int m_db_size) = 0;

        virtual Scalar get_w_delta(int layer_n, int i, Scalar m_dw_i) = 0;

        virtual Scalar get_b_delta(int layer_n, int i, Scalar m_dw_i) = 0;
    };


} // namespace MiniDNN


#endif /* OPTIMIZER_H_ */
