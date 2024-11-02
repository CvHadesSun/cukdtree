#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "cukd/box.h"
#include "warpper/cukd_tree.h"

namespace py = pybind11;
using namespace cukd;
using namespace cukd::common;

template <typename point_t>
void bind_box_t(py::module& m) {
  py::class_<box_t<point_t>>(m, "Box")
      .def(py::init<>())  // 默认构造函数
      .def_readwrite("lower", &box_t<point_t>::lower)
      .def_readwrite("upper", &box_t<point_t>::upper)
      .def("contains", &box_t<point_t>::contains,
           "Check if point is within the bounding box")
      .def("grow", &box_t<point_t>::grow,
           "Expand the bounding box to include the given point")
      .def("setEmpty", &box_t<point_t>::setEmpty,
           "Set the bounding box to an empty state")
      .def("setInfinite", &box_t<point_t>::setInfinite,
           "Set the bounding box to an infinite open state")
      .def("widestDimension", &box_t<point_t>::widestDimension,
           "Get the dimension with the widest extent")
      .def("__repr__", [](const box_t<point_t>& b) {
        return "<Box lower=" + std::to_string(b.lower.x) +
               ", upper=" + std::to_string(b.upper.x) + ">";
      });
}

PYBIND11_MODULE(cuda_kdtree, m) {
  bind_box_t<float3>(m);
  m.def("build_kdtree", &build_kdtree, py::return_value_policy::reference);
  m.def("query", &query, py::return_value_policy::reference);
}

// PYBIND11_MODULE(cuda_kdtree, m) {
//     m.def("build_kdtree", &py_build_kdtree, "Build a KD-tree using CUDA and
//     return bounds as a Python list");
// }