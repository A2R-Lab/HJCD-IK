#pragma once
#include <string>
#include "src/collision/environment.hh"

namespace pRRTC {

struct EnvCache {
    ppln::collision::Environment<float> h_env{};
    ppln::collision::Environment<float>* d_env = nullptr;
    std::string key;
    bool ready = false;
};

}
