#include "../../include/sedona/sedona.h"
#include <iostream>
#include <random>
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

#include <chrono>

int main() {

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

   // MySystem sys;
    //sys.set({1, 2, 3}, {0, 0, 0});
    std::cout << sys.pos() << std::endl;
    std::cout << sys.accel() << std::endl;
    float h = 0.01;
    float runtime = 100;
    auto t1 = std::chrono::high_resolution_clock::now();
    sys.simulate(runtime, h);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << sys.pos() << std::endl; // Keep the optimizer from removing our simulation entirely
    std::cout << "Average single iteration speed " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() / (runtime / h) << " microseconds. "<< std::endl;

    return 0;

}