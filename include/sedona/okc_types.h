//
// Created by danie on 11/17/2022.
//

#ifndef SEDONA_OKC_TYPES_H
#define SEDONA_OKC_TYPES_H

#include <Eigen/Eigen>
#include <array>

namespace sedona {
    template<class T, int N>
    struct Frame {

        Eigen::Matrix3 <T> rotation = Eigen::Matrix3<T>::Identity();
        Eigen::Vector3 <T> position = Eigen::Vector3<T>::Zero();

        Eigen::Vector3 <T> ang_vel = Eigen::Vector3<T>::Zero();
        Eigen::Vector3 <T> lin_vel = Eigen::Vector3<T>::Zero();

        Eigen::Matrix<T, 6, 6> jacobian_pmap = Eigen::Matrix<T, 6, 6>::Identity();
        Eigen::Matrix<T, 6, 6> dot_jacobian_pmap = Eigen::Matrix<T, 6, 6>::Zero();

        Eigen::Matrix<T, 6, N> jacobian = Eigen::Matrix<T, 6, N>::Zero();
        Eigen::Matrix<T, 6, N> dot_jacobian = Eigen::Matrix<T, 6, N>::Zero();
    };

    template<class T>
    struct MassSpec {
        const T mass;
        const Eigen::Vector3 <T> com;
        const Eigen::Matrix3 <T> inertia;
    };

    template<class T, int N>
    struct Dynamics_State {

        Eigen::Matrix <T, N, N> mass_matrix;
        Eigen::Vector <T, N> centrifugal;
        Eigen::Vector <T, N> forces;

        static Dynamics_State Zero() {
            return Dynamics_State<T, N>{
                    .mass_matrix = Eigen::Matrix<T, N, N>::Zero(),
                    .centrifugal = Eigen::Vector<T, N>::Zero(),
                    .forces = Eigen::Vector<T, N>::Zero(),
            };
        }

        Dynamics_State operator+(Dynamics_State rhs) {
            return Dynamics_State<T, N>{
                    .mass_matrix = this->mass_matrix + rhs.mass_matrix,
                    .centrifugal = this->centrifugal + rhs.centrifugal,
                    .forces = this->forces + rhs.forces,
            };
        }

        void operator+=(Dynamics_State rhs) {
            mass_matrix += rhs.mass_matrix;
            centrifugal += rhs.centrifugal;
            forces += rhs.centrifugal;
        }

        template<int N_RHS>
        void operator+=(Dynamics_State<T, N_RHS> rhs) {
            static_assert(N_RHS <= N,
                          "YOUR LINK HAS MORE DIMENSIONS THAN THE ENTIRE SYSTEM");
            mass_matrix.template block<N_RHS, N_RHS>(0, 0) += rhs.mass_matrix;
            centrifugal.template block<N_RHS, 1>(0, 0) += rhs.centrifugal;
            forces.template block<N_RHS, 1>(0, 0) += rhs.forces;
        }
    };

    template<class T, int N>
    struct LinkDynamicsState : public Dynamics_State<T, N> {

        Eigen::Matrix<T, 6, 6> local_mass_matrix;
        Eigen::Vector<T, 6> d_link_star;
    };
}

#endif // SEDONA_OKC_TYPES_H
