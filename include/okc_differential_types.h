#ifndef SEDONA_OKC_DIFFERENTIAL_TYPES_H
#define SEDONA_OKC_DIFFERENTIAL_TYPES_H

#include "okc_math.h"
#include "okc_types.h"

namespace sedona {
    template<class T, size_t N>
    struct DifferentialFrame {

        Tensor<T, N, 3, 3> rotation = Tensor_Zero<T, N, 3, 3>();
        Eigen::Matrix<T, 3, N> position = Eigen::Matrix<T, 3, N>::Zero();

        Eigen::Matrix<T, 3, N> ang_vel = Eigen::Matrix<T, 3, N>::Zero();
        Eigen::Matrix<T, 3, N> lin_vel = Eigen::Matrix<T, 3, N>::Zero();

        Tensor<T, N, 6, 6> jacobian_pmap = Tensor_Zero<T, N, 6, 6>();
        Tensor<T, N, 6, 6> dot_jacobian_pmap = Tensor_Zero<T, N, 6, 6>();

        Tensor<T, N, 6, N> jacobian = Tensor_Zero<T, N, 6, N>();
        Tensor<T, N, 6, N> dot_jacobian = Tensor_Zero<T, N, 6, N>();
    };

    template<class T, size_t N>
    struct Differential_Dynamics_State {

        Tensor<T, N, N, N> mass_matrix;
        Eigen::Matrix <T, N, N> centrifugal;
        Eigen::Matrix <T, N, N> forces;

        static Differential_Dynamics_State Zero() {
            return Differential_Dynamics_State<T, N>{
                    .mass_matrix = Tensor_Zero<T, N, N, N>(),
                    .centrifugal = Eigen::Matrix<T, N, N>::Zero(),
                    .forces = Eigen::Matrix<T, N, N>::Zero(),
            };
        }

        Differential_Dynamics_State operator+(Differential_Dynamics_State rhs) {
            return Dynamics_State<T, N>{
                    .mass_matrix = this->mass_matrix + rhs.mass_matrix,
                    .centrifugal = this->centrifugal + rhs.centrifugal,
                    .forces = this->forces + rhs.forces,
            };
        }

        void operator+=(Differential_Dynamics_State rhs) {
            mass_matrix += rhs.mass_matrix;
            centrifugal += rhs.centrifugal;
            forces += rhs.centrifugal;
        }
    };

    template<class T, size_t N>
    void differential_kinematics_(const Frame<T, N> &parent_g,
                                  const DifferentialFrame<T, N> &d_parent_g,
                                  const Frame<T, N> &frame_l,
                                  const DifferentialFrame<T, N> &d_frame_l,
                                  const Frame<T, N> &frame_g,
                                  DifferentialFrame<T, N> &d_frame_g,
                                  size_t idx = N) {

        d_frame_g.rotation = d_parent_g.rotation * frame_l.rotation +
                             parent_g.rotation * d_frame_l.rotation;
        d_frame_g.position = d_parent_g.position +
                             tvp(d_parent_g.rotation, frame_l.position) +
                             parent_g.rotation * d_frame_l.position;

        d_frame_g.lin_vel = d_parent_g.lin_vel +
                            tvp(skew3(d_parent_g.ang_vel), frame_g.position) +
                            skew(parent_g.ang_vel) * d_frame_g.position;
        d_frame_g.ang_vel = d_parent_g.ang_vel +
                            tvp(d_frame_g.rotation, frame_l.ang_vel) +
                            frame_g.rotation * d_frame_l.ang_vel;

        for (size_t i = 0; i <= idx; ++i) {

            d_frame_g.jacobian_pmap[i].template block<3, 3>(0, 0) =
                    d_frame_l.rotation[i].transpose();
            d_frame_g.jacobian_pmap[i].template block<3, 3>(3, 0) =
                    -d_parent_g.rotation[i] * skew(frame_l.position) -
                    parent_g.rotation * skew(Eigen::Vector3<T>(d_frame_l.position.col(i)));

            Eigen::Matrix<T, 6, N> jac_vec;
            jac_vec << d_frame_l.jacobian[i].template topRows<3>(),
                    d_parent_g.rotation[i] * frame_l.jacobian.template bottomRows<3>() +
                    parent_g.rotation * d_frame_l.jacobian[i].template bottomRows<3>();

            d_frame_g.jacobian[i] = d_frame_g.jacobian_pmap[i] * parent_g.jacobian +
                                    frame_g.jacobian_pmap * d_parent_g.jacobian[i] +
                                    jac_vec;

            d_frame_g.dot_jacobian_pmap[i].template block<3, 3>(0, 0) =
                    -skew(Eigen::Vector3<T>(d_frame_l.ang_vel.col(i))) *
                    frame_l.rotation.transpose() -
                    skew(frame_l.ang_vel) * d_frame_l.rotation[i].transpose();

            d_frame_g.dot_jacobian_pmap[i].template block<3, 3>(3, 0) =
                    -d_parent_g.rotation[i] *
                    (skew(parent_g.ang_vel) * skew(frame_l.position) +
                     skew(frame_l.lin_vel)) +
                    -parent_g.rotation *
                    (skew(Eigen::Vector3<T>(d_parent_g.ang_vel.col(i))) *
                     skew(frame_l.position) +
                     skew(parent_g.ang_vel) *
                     skew(Eigen::Vector3<T>(d_frame_l.position.col(i))) +
                     skew(Eigen::Vector3<T>(d_frame_l.lin_vel.col(i))));

            Eigen::Matrix<T, 6, N> d_jac_vec;

            d_jac_vec << Eigen::Matrix<T, 3, N>::Zero(),
                    (d_parent_g.rotation[i] * skew(parent_g.ang_vel) +
                     parent_g.rotation * skew(Eigen::Vector3<T>(d_parent_g.ang_vel.col(i)))) *
                    frame_l.jacobian.template bottomRows<3>();

            d_frame_g.dot_jacobian[i] =
                    d_frame_g.dot_jacobian_pmap[i] * parent_g.jacobian +
                    frame_g.dot_jacobian_pmap * d_parent_g.jacobian[i] +
                    d_frame_g.jacobian_pmap[i] * parent_g.dot_jacobian +
                    frame_g.jacobian_pmap * d_parent_g.dot_jacobian[i] + d_jac_vec;
        }
    }

    template<class T, size_t N>
    void differential_dynamics_(const MassSpec<T> &mass_spec,
                                const Frame<T, N> &frame_g,
                                const DifferentialFrame<T, N> &d_frame_g,
                                const LinkDynamicsState<T, N> &dynamics_state,
                                Differential_Dynamics_State<T, N> &d_dynamics_state,
                                const Eigen::Vector <T, N> &d_x,
                                const Eigen::Matrix <T, N, N> &d_d_x, int idx = N) {

        const T mass = mass_spec.mass;
        const Eigen::Vector3 <T> com = mass_spec.com;
        const Eigen::Vector3 <T> mass_com = mass * com;
        const Eigen::Matrix3 <T> inertia = mass_spec.inertia;

        for (size_t i = 0; i <= idx; ++i) {
            Eigen::Matrix<T, 3, 3> d_m_corner =
                    skew(mass_spec.com) * d_frame_g.rotation[i].transpose();

            Eigen::Matrix<T, 6, 6> d_local_mass_matrix;
            d_local_mass_matrix << Eigen::Matrix3<T>::Zero(), d_m_corner,
                    d_m_corner.transpose(), Eigen::Matrix3<T>::Zero();

            d_dynamics_state.mass_matrix[i] =
                    d_frame_g.jacobian[i].transpose() * dynamics_state.local_mass_matrix *
                    frame_g.jacobian +
                    frame_g.jacobian.transpose() *
                    (d_local_mass_matrix * frame_g.jacobian +
                     dynamics_state.local_mass_matrix * d_frame_g.jacobian[i]);

            Eigen::Vector<T, 6> d_d_link_star;

            d_d_link_star << skew(Eigen::Vector3<T>(d_frame_g.ang_vel.col(i))) *
                             (inertia * frame_g.ang_vel) +
                             skew(frame_g.ang_vel) *
                             (inertia * d_frame_g.ang_vel.col(i)),
                    (d_frame_g.rotation[i] *
                     frame_g.ang_vel.cross(frame_g.ang_vel.cross(mass_com))) +
                    (frame_g.rotation * 2 * skew(frame_g.ang_vel) *
                     skew(Eigen::Vector3<T>(d_frame_g.ang_vel.col(i))) * mass_com);

            d_dynamics_state.centrifugal.col(i) =
                    d_frame_g.jacobian[i].transpose() *
                    (dynamics_state.local_mass_matrix * frame_g.dot_jacobian * d_x +
                     dynamics_state.d_link_star) +
                    frame_g.jacobian.transpose() *
                    (d_local_mass_matrix * frame_g.dot_jacobian * d_x +
                     dynamics_state.local_mass_matrix *
                     (d_frame_g.dot_jacobian[i] * d_x +
                      frame_g.dot_jacobian * d_d_x.col(i)) +
                     d_d_link_star);
        }
    }
}

#endif // SEDONA_OKC_DIFFERENTIAL_TYPES_H
