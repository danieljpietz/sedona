#include "../../include/sedona/sedona.h"
#include "codegen/doublePendulumDerivative.h"
#include <random>
#include <cassert>
#include <iostream>

using namespace sedona;

class MySystem : public OKC<double, 3> {

public:

    using Link = Link<double, 3>;

    Link x = Link {
            .link_type = Revolute,
            .axis = {1, 0, 0},
            .offset = {
                    .position = {0, 0, 0},
            },
            .mass_spec = {
                    .mass = 1,
                    .com = {0, 0.5, 0},
                    .inertia = Eigen::Matrix3d::Identity(),
            }
    };

    Link y = Link {
            .parent = &x,
            .link_type = Prismatic,
            .axis = {0, 1, 0},
            .offset = {
                    .position = {0, 1, 0},
            },
            .mass_spec = {
                    .mass = 1,
                    .com = {0, 0.5, 0},
                    .inertia = Eigen::Matrix3d::Identity(),
            }
    };

    Link z = Link {
            .parent = &y,
            .link_type = Revolute,
            .axis = {1, 0, 0},
            .offset = {
                    .position = {0, 1, 0},
            },
            .mass_spec = {
                    .mass = 1,
                    .com = {0, 0.5, 0},
                    .inertia = Eigen::Matrix3d::Identity(),
            }
    };

    MySystem() {
        set_links({&x, &y, &z});
    }


};

int main() {

    // Generate a random configuration for the system

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 4);

    double theta_1 = dist(gen);
    double theta_2 = dist(gen);
    double theta_3 = dist(gen);
    double dot_theta_1 = dist(gen);
    double dot_theta_2 = dist(gen);
    double dot_theta_3 = dist(gen);

    // Do the computation using sedona
    MySystem sys;
    sys.set({theta_1, theta_2, theta_3}, {dot_theta_1, dot_theta_2, dot_theta_3});
    auto accel = sys.d_accel();

    // Do the computation using the generated symbolic code from MATLAB
    double d_xddot[36];
    doublePendulumDerivative(dot_theta_1, dot_theta_2, dot_theta_3,
                             theta_2, theta_3, d_xddot);

    // Compare the results
    for (int i = 0; i < 36; ++i)
        assert(std::abs(d_xddot[i] - accel.data()[i]) < 1e-6);

    return 0;

}