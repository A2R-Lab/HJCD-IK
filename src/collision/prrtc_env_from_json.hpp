#pragma once
#include <string>
#include <nlohmann/json.hpp>
#include "src/collision/environment.hh"

namespace pRRTC {

ppln::collision::Environment<float>
problem_dict_to_env(const nlohmann::json& problem, const std::string& name);

// frees arrays allocated with new[]
void free_host_env(ppln::collision::Environment<float>& env);

// pick problem instance from json string
nlohmann::json
select_problem_instance(const nlohmann::json& problems_root,
                        const std::string& problem_set_name,
                        int problem_idx);

}
