/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: phillicl

/*! \file FENEBondForceComputeGPU.cc
    \brief Defines the FENEBondForceComputeGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include "FENEBondForceComputeGPU.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \param sysdef System to compute bond forces on
    \param log_suffix Name given to this instance of the fene bond
*/
FENEBondForceComputeGPU::FENEBondForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, const std::string& log_suffix)
        : FENEBondForceCompute(sysdef, log_suffix), m_block_size(64)
    {
    // only one GPU is currently supported
    if (!exec_conf.isCUDAEnabled())
        {
        cerr << endl 
             << "***Error! Creating a FENEBondForceComputeGPU with no GPU in the execution configuration" 
             << endl << endl;
        throw std::runtime_error("Error initializing FENEBondForceComputeGPU");
        }
        
    // allocate and zero device memory for K, R0 parameters
    cudaMalloc(&m_gpu_params, m_bond_data->getNBondTypes()*sizeof(float4));
    cudaMemset(m_gpu_params, 0, m_bond_data->getNBondTypes()*sizeof(float4));
    CHECK_CUDA_ERROR();
    
    // allocate host memory for GPU parameters
    m_host_params = new float4[m_bond_data->getNBondTypes()];
    memset(m_host_params, 0, m_bond_data->getNBondTypes()*sizeof(float4));
    
    // allocate device memory for the radius error check parameters
    cudaMalloc(&m_checkr, sizeof(int));
    cudaMemset(m_checkr, 0, sizeof(int));
    CHECK_CUDA_ERROR();
    }

FENEBondForceComputeGPU::~FENEBondForceComputeGPU()
    {
    // free memory on the GPU
    cudaFree(m_gpu_params);
    m_gpu_params = NULL;
        
    cudaFree(m_checkr);
    m_checkr = NULL;
    CHECK_CUDA_ERROR();
    
    // free memory on the CPU
    delete[] m_host_params;
    m_host_params = NULL;
    }

/*! \param type Type of the bond to set parameters for
    \param K Stiffness parameter for the force computation
    \param r_0 Equilibrium length for the force computation
    \param sigma  Particle diameter
    \param epsilon Determines hardness of the particles in the WCA part of the interaction (usually set to 1/T)
s
    Sets parameters for the potential of a particular bond type and updates the
    parameters on the GPU.
*/
void FENEBondForceComputeGPU::setParams(unsigned int type, Scalar K, Scalar r_0, Scalar sigma, Scalar epsilon)
    {
    FENEBondForceCompute::setParams(type, K, r_0, sigma, epsilon);
    
    // update the local copy of the memory
    m_host_params[type] = make_float4(K, r_0, sigma, epsilon);
    
    // copy the parameters to the GPU
    cudaMemcpy(m_gpu_params, m_host_params, m_bond_data->getNBondTypes()*sizeof(float4), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();
    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_fene_bond_forces to do the dirty work.
*/
void FENEBondForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start the profile
    if (m_prof) m_prof->push(exec_conf, "FENE");
    
    gpu_bondtable_array& gpu_bondtable = m_bond_data->acquireGPU();
    
    // the bond table is up to date: we are good to go. Call the kernel
    gpu_pdata_arrays& pdata = m_pdata->acquireReadOnlyGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
    
    // hackish method for tracking exceedsR0 over multiple GPUs
    unsigned int exceedsR0;
        
    // run the kernel
    exceedsR0 = 0;
    gpu_compute_fene_bond_forces(m_gpu_forces.d_data,
                                 pdata,
                                 box,
                                 gpu_bondtable,
                                 m_gpu_params,
                                 m_checkr,
                                 m_bond_data->getNBondTypes(),
                                 m_block_size,
                                 exceedsR0);
    if (exec_conf.isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
    if (exceedsR0)
        {
        cerr << endl << "***Error! FENE bond length exceeds maximum permitted" << endl << endl;
        throw std::runtime_error("Error in fene bond calculation");
        }
        
    // the force data is now only up to date on the gpu
    m_data_location = gpu;
    
    m_pdata->release();
    
    int64_t mem_transfer = m_pdata->getN() * 4+16+20 + m_bond_data->getNumBonds() * 2 * (8+16+8);
    int64_t flops = m_bond_data->getNumBonds() * 2 * (3+12+8+6+7+3+7);
    if (m_prof) m_prof->pop(exec_conf, flops, mem_transfer);
    }

void export_FENEBondForceComputeGPU()
    {
    class_<FENEBondForceComputeGPU, boost::shared_ptr<FENEBondForceComputeGPU>, bases<FENEBondForceCompute>, boost::noncopyable >
    ("FENEBondForceComputeGPU", init< boost::shared_ptr<SystemDefinition>, const std::string& >())
    .def("setBlockSize", &FENEBondForceComputeGPU::setBlockSize)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

