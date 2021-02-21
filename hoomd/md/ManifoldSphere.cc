// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "ManifoldSphere.h"
#include <pybind11/pybind11.h>

//! Exports the Sphere manifold class to python
void export_ManifoldSphere(pybind11::module& m)
    {
    pybind11::class_< ManifoldSphere, std::shared_ptr<ManifoldSphere> >(m, "ManifoldSphere")
    .def(pybind11::init<Scalar, Scalar3 >())
    .def("implicit_function", &ManifoldSphere::implicit_function)
    .def("derivative", &ManifoldSphere::derivative)
    ;
    }
