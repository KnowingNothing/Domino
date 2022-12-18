#include <ir_base.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace domino {

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(domino, m) {
    m.def("add", &add, "A function for add.");
}

}  // namespace domino