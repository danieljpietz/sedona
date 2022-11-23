#ifndef SEDONA_OKC_SYSTEM_H
#define SEDONA_OKC_SYSTEM_H

#include "okc_link.h"
#include "okc_types.h"
#include <Eigen/Eigen>
#include <map>

namespace sedona {

    template<class T, int N, class STORAGE_CLASS>
    class OKCBase {
    protected:

        int dof = N;
        T _time = 0;
        STORAGE_CLASS _links;

        Eigen::Vector <T, N> _x;
        Eigen::Vector <T, N> _dot_x;
        Eigen::Vector <T, N> _ddot_x;
        Eigen::Matrix <T, N, N> _dx_ddot_x;
        Eigen::Matrix <T, N, N> _ddx_ddot_x;

        Dynamics_State<T, N> _dynamics_state;

        Differential_Dynamics_State <T, N> dx_dynamics_state;
        Differential_Dynamics_State <T, N> ddx_dynamics_state;

        OKCBase() {}

        virtual void set_links(STORAGE_CLASS links) {
            for (int i = 0; i < dof; ++i) {
                Link<T, N> *link = links[i];
                link->idx = i;
                link->init();
                _links[i] = links[i];
            }
        }

        void reset_dynamics() {
            _dynamics_state = Dynamics_State<T, N>::Zero();
            dx_dynamics_state = Differential_Dynamics_State<T, N>::Zero();
            ddx_dynamics_state = Differential_Dynamics_State<T, N>::Zero();
        }

        virtual void _link_set() {
            for (auto link: _links) {
                link->set(_x, _dot_x);
                _dynamics_state += link->_dynamics_state;
                dx_dynamics_state += link->dx_dynamics_state;
                ddx_dynamics_state += link->ddx_dynamics_state;
            }
        }

        void _collect_dynamics() {
            Eigen::Matrix <T, N, N> system_mass_inverse =
                    _dynamics_state.mass_matrix.inverse();

            _ddot_x = system_mass_inverse * (-_dynamics_state.centrifugal);

            for (int i = 0; i < N; ++i) {
                _dx_ddot_x.col(i) =
                        system_mass_inverse * (-dx_dynamics_state.centrifugal.col(i) -
                                               dx_dynamics_state.mass_matrix[i] * _ddot_x);
                _ddx_ddot_x.col(i) =
                        system_mass_inverse * (-ddx_dynamics_state.centrifugal.col(i) -
                                               ddx_dynamics_state.mass_matrix[i] * _ddot_x);
            }
        }

        virtual void _update() {
            reset_dynamics();
            _link_set();
            _collect_dynamics();
            update();
        }

    public:

        inline const Eigen::Vector <T, N> &pos() const { return _x; }

        inline const Eigen::Vector <T, N> &vel() const { return _dot_x; }

        inline const Eigen::Vector <T, N> &accel() const { return _ddot_x; }

        inline Eigen::Matrix<T, 2 * N, 2 * N> d_accel() const {
            Eigen::Matrix<T, 2 * N, 2 * N> ret;
            ret << Eigen::Matrix<T, N, N>::Zero(), Eigen::Matrix<T, N, N>::Identity(), _dx_ddot_x, _ddx_ddot_x;
            return ret;
        }

        inline const T time() const { return _time; }

        inline const Dynamics_State<T, N> &dynamics_state() const {
            return _dynamics_state;
        }

        void set(const Eigen::Vector <T, N> &joint_positions,
                 const Eigen::Vector <T, N> &joint_velocities) {
            _x = joint_positions;
            _dot_x = joint_velocities;
            _update();
        }

        void step(T step_size) {
            _dot_x += step_size * this->_ddot_x;
            _x += step_size * _dot_x;
            _update();
        }

        std::map <std::string, Eigen::Vector<T, Eigen::Dynamic>>
        simulate(T runtime, T step_size) {

            int len = runtime / step_size;
            Eigen::Matrix <T, N, Eigen::Dynamic> positions =
                    Eigen::Matrix<T, N, Eigen::Dynamic>(N, len);
            Eigen::Matrix <T, N, Eigen::Dynamic> velocities =
                    Eigen::Matrix<T, N, Eigen::Dynamic>(N, len);

            for (int i = 0; i < len; ++i) {
                _time += step_size;
                positions.col(i) = this->pos();
                velocities.col(i) = this->vel();
                this->step(step_size);
            }

            std::map <std::string, Eigen::Vector<T, Eigen::Dynamic>> result;

            for (auto link: _links) {
                std::string col_name =
                        link->name.length() ? link->name : std::to_string(link->idx);
                Eigen::Vector <T, Eigen::Dynamic> link_positions =
                        Eigen::Vector<T, Eigen::Dynamic>(len);
                Eigen::Vector <T, Eigen::Dynamic> link_velocities =
                        Eigen::Vector<T, Eigen::Dynamic>(len);
                link_positions = positions.row(link->idx);
                link_velocities = velocities.row(link->idx);
                result[col_name] = link_positions;
                result["v " + col_name] = link_positions;
            }

            return result;
        }

        virtual void update() {}
    };

    template<class T, int N>
    using OKC = OKCBase<T, N, std::array<Link<T, N> *, N>>;

    template<class T>
    using DynamicOKC = OKCBase<T, Eigen::Dynamic, std::vector<Link<T, Eigen::Dynamic> *>>;

}

#endif // SEDONA_OKC_SYSTEM_H
