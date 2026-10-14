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
            // return 1.;

            // if (tau > this->num_threads * 1.125) {
            //     return 0.;
            // }
            if (this->tau <= std::sqrt(num_threads)) {
                return 2;
            }
            else if (std::sqrt(num_threads) <= this->tau && this->tau <= num_threads / 2) {
                return 1;
            }
            else if (num_threads / 2 <= this->tau && this->tau <= num_threads) {
                return 0.5;
            }
            else {
                return 0.2;
            }
        }

    public:
        Scalar m_lrate;

        float step_scale_factor = 1.0;

        long tau = 0;
        int num_threads = 0;

        Optimizer(int num_threads) {
            this->num_threads = num_threads;
        }

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
