#ifndef SEDONA_MT_OKC_H
#define SEDONA_MT_OKC_H

#include "okc_types.h"
#include "okc_system.h"
#include "okc_link.h"

#include <array>
#include <thread>
#include <barrier>
#include <functional>
#include <mutex>
#include <condition_variable>

template<class T, size_t N>
class MT_OKC : public OKC<T, N> {

    std::array<std::thread, 6 * N> _threads;

    std::barrier<> stage1_start_barrier = std::barrier(N + 1);
    std::barrier<> stage1_stop_barrier = std::barrier(N + 1);

    std::barrier<> stage2_start_barrier = std::barrier(N + 1);
    std::barrier<> stage2_stop_barrier = std::barrier(N + 1);

    std::barrier<> stage3_start_barrier = std::barrier(N + 1);
    std::barrier<> stage3_stop_barrier = std::barrier(N + 1);

    std::barrier<> stage4_start_barrier = std::barrier(N + 1);
    std::barrier<> stage4_stop_barrier = std::barrier(N + 1);

    [[noreturn]] void _local_kinematics_worker(Link<T, N> *link) {
        while (true) {
            stage1_start_barrier.arrive_and_wait(); // Wait for parent to start us
            link->local_kinematics(this->_x, this->_dot_x);
            auto token = stage1_stop_barrier.arrive(); // Signal to parent that we're done
        }
    }

    [[noreturn]] void _kinematics_worker(Link<T, N> *link) {
        while (true) {
            stage2_start_barrier.arrive_and_wait(); // Wait for parent to start us
            link->kinematics();
            auto token = stage2_stop_barrier.arrive(); // Signal to parent that we're done
        }
    }

    [[noreturn]] void _local_differential_kinematics_worker(Link<T, N> *link) {
        while (true) {
            stage2_start_barrier.arrive_and_wait(); // Wait for parent to start us
            link->local_differential_kinematics(this->_x, this->_dot_x);
            auto token = stage2_stop_barrier.arrive(); // Signal to parent that we're done
        }
    }

    [[noreturn]] void _differential_kinematics_worker(Link<T, N> *link) {
        while (true) {
            stage3_start_barrier.arrive_and_wait(); // Wait for parent to start us
            link->differential_kinematics();
            auto token = stage3_stop_barrier.arrive(); // Signal to parent that we're done
        }
    }

    [[noreturn]] void _dynamics_worker(Link<T, N> *link) {
        while (true) {
            stage3_start_barrier.arrive_and_wait(); // Wait for parent to start us
            link->dynamics(this->_dot_x);
            auto token = stage3_stop_barrier.arrive(); // Signal to parent that we're done
        }
    }

    [[noreturn]] void _differential_dynamics_worker(Link<T, N> *link) {
        while (true) {
            stage4_start_barrier.arrive_and_wait(); // Wait for parent to start us
            link->differential_dynamics(this->_dot_x);
            auto token = stage4_stop_barrier.arrive(); // Signal to parent that we're done
        }
    }

    void _link_set() override {

        // STAGE 1 - Local Kinematics
        auto token = stage1_start_barrier.arrive();
        stage1_stop_barrier.arrive_and_wait();

        // STAGE 2 - Local Differential Kinematics and Kinematics
        token = stage2_start_barrier.arrive();
        for (auto &link: this->_links)
            link->kinematics();
        stage2_stop_barrier.arrive_and_wait();

        // STAGE 3 - Differential Kinematics and Dynamics
        token = stage3_start_barrier.arrive();
        for (auto &link: this->_links)
            link->differential_kinematics();
        stage3_stop_barrier.arrive_and_wait();

        // STAGE 4 - Differential Dynamics
        token = stage4_start_barrier.arrive();
        stage4_stop_barrier.arrive_and_wait();

        for (auto link: this->_links) {
            this->_dynamics_state += link->_dynamics_state;
            this->dx_dynamics_state += link->dx_dynamics_state;
            this->ddx_dynamics_state += link->ddx_dynamics_state;
        }

    }

public:

    MT_OKC() : OKC<T, N>() {}

    virtual void set_links(std::array<Link<T, N> *, N> links) override {
        OKC<T, N>::set_links(links);

        size_t thread_index = 0;

        for (auto link: this->_links) {
            _threads[thread_index++] = std::thread(&MT_OKC::_local_kinematics_worker, this, link);

            _threads[thread_index++] = std::thread(&MT_OKC::_local_differential_kinematics_worker, this, link);
            //_threads[thread_index++] = std::thread(&MT_OKC::_kinematics_worker, this, link);

            _threads[thread_index++] = std::thread(&MT_OKC::_dynamics_worker, this, link);
            //_threads[thread_index++] = std::thread(&MT_OKC::_differential_kinematics_worker, this, link);

            _threads[thread_index++] = std::thread(&MT_OKC::_differential_dynamics_worker, this, link);
        }
    }


};

template<class T, size_t N>
class MT_Link : public Link<T, N> {

public:

};

#endif //SEDONA_MT_OKC_H
