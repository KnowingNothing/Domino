#include <pybind11/pybind11.h>
#include <type_system/dtype.h>

namespace py = pybind11;

namespace domino {

PYBIND11_MODULE(dominoc, m) {
  /// bind type_system
  py::enum_<DTypeKind>(m, "DTypeKind")
      .value("Int", DTypeKind::kInt)
      .value("UInt", DTypeKind::kUInt)
      .value("Float", DTypeKind::kFloat)
      .value("BFloat", DTypeKind::kBFloat)
      .value("TFloat", DTypeKind::kTFloat)
      .value("MemRef", DTypeKind::kMemRef)
      .value("String", DTypeKind::kString)
      .value("Complex", DTypeKind::kComplex)
      .value("IGNORE", DTypeKind::kIGNORE)
      .value("UNKNOWN", DTypeKind::kUNKNOWN);
  py::class_<DType>(m, "DType")
      .def(py::init<DTypeKind, int, int>())
      .def("is_int", &DType::is_int)
      .def("is_uint", &DType::is_uint)
      .def("is_float", &DType::is_float)
      .def("is_bfloat", &DType::is_bfloat)
      .def("is_tfloat", &DType::is_tfloat)
      .def("is_memref", &DType::is_memref)
      .def("is_string", &DType::is_string)
      .def("is_complex", &DType::is_complex)
      .def("is_ignore", &DType::is_ignore)
      .def("is_unknown", &DType::is_unknown)
      .def("copy", &DType::copy)
      .def("with_lanes", &DType::with_lanes)
      .def_readonly("type_kind", &DType::type_kind)
      .def_readonly("bits", &DType::bits)
      .def_readonly("lane", &DType::lane);
}

}  // namespace domino