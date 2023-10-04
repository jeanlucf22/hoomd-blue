// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPair.h"
#include "EvaluatorPairTwoQ.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
template void export_AnisoPotentialPair<EvaluatorPairTwoQ>(pybind11::module& m,
                                                         const std::string& name);

void export_AnisoPotentialPairTwoQ(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairTwoQ>(m, "AnisoPotentialPairTwoQ");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
