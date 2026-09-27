#include "kernel/problem_document.h"
#include <cassert>
#include <string>

template <class Fn>
void must_throw(Fn fn) {
    bool threw = false;
    try { fn(); } catch (const std::exception&) { threw = true; }
    assert(threw);
}

int main() {
    hjcd_env::ProblemDocument document;
    const std::string first = R"({"problems":{"a":[{"id":1},{"id":2}],"b":[{"id":3}]}})";
    assert(document.update(first));
    const auto* selected = &document.select("a", 0);
    assert(selected->at("id") == 1);
    assert(document.select("a", 1).at("id") == 2);
    assert(document.select("b", 0).at("id") == 3);
    const std::string separate_storage(first);
    assert(!document.update(separate_storage));
    assert(&document.select("a", 0) == selected); // parsed tree reused, no scene copies
    must_throw([&] { document.update("{invalid"); });
    assert(!document.update(first)); // failed parse preserves prior identity and data
    assert(&document.select("a", 0) == selected);
    must_throw([&] { document.select("missing", 0); });
    must_throw([&] { document.select("a", -1); });
    must_throw([&] { document.select("a", 2); });
    std::string changed(first);
    changed[changed.find("id\":1") + 4] = '9'; // same length, distinct content
    assert(document.update(changed));
    assert(document.select("a", 0).at("id") == 9);
    assert(document.update(R"({"problems":{"a":{}}})"));
    must_throw([&] { document.select("a", 0); });
    assert(document.update(first));
    assert(document.select("a", 0).at("id") == 1);
}
