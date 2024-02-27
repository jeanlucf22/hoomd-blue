// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __EVALUATOR_PAIR_TWOQ_H__
#define __EVALUATOR_PAIR_TWOQ_H__

#ifndef __HIPCC__
#include <string>
#endif

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include "hoomd/VectorMath.h"
#include "hoomd/extern/qtwo/src/Quaternion.h"
#include "hoomd/extern/qtwo/src/PotentialPrism.h"

/*! \file EvaluatorPairTwoQ.h
    \brief Defines a an evaluator class for the Gay-Berne potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host
//! compiler
#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {
/*!
 * Gay-Berne potential as formulated by Allen and Germano,
 * with shape-independent energy parameter, for identical uniaxial particles.
 */

class EvaluatorPairTwoQ
    {
    public:
    struct param_type
        {
        Scalar epsilon; //! The energy scale.
        Scalar lperp;   //! The semiaxis length perpendicular to the particle orientation.
        Scalar lpar;    //! The semiaxis length parallel to the particle orientation.

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory
            allocation
        */
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        //! Set CUDA memory hints
        void set_memory_hint() const
            {
            // default implementation does nothing
            }
#endif

        HOSTDEVICE param_type()
            {
            epsilon = 0;
            lperp = 0;
            lpar = 0;
            }

#ifndef __HIPCC__

        param_type(pybind11::dict v, bool managed = false)
            {
            epsilon = v["epsilon"].cast<Scalar>();
            lperp = v["lperp"].cast<Scalar>();
            lpar = v["lpar"].cast<Scalar>();
            }

        pybind11::dict toPython()
            {
            pybind11::dict v;
            v["epsilon"] = epsilon;
            v["lperp"] = lperp;
            v["lpar"] = lpar;
            return v;
            }

#endif
        }
#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    // Nullary structure required by AnisoPotentialPair.
    struct shape_type
        {
        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory allocation
        */
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

        HOSTDEVICE shape_type() { }

#ifndef __HIPCC__

        shape_type(pybind11::object shape_params, bool managed) { }

        pybind11::object toPython()
            {
            return pybind11::none();
            }
#endif

#ifdef ENABLE_HIP
        //! Attach managed memory to CUDA stream
        void set_memory_hint() const { }
#endif
        };

    //! Constructs the pair potential evaluator
    /*! \param _dr Displacement vector between particle centers of mass
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _q_i Quaternion of i^th particle
        \param _q_j Quaternion of j^th particle
        \param _params Per type pair parameters of this potential
    */
    HOSTDEVICE EvaluatorPairTwoQ(const Scalar3& _dr,
                               const Scalar4& _qi,
                               const Scalar4& _qj,
                               const Scalar _rcutsq,
                               const param_type& _params)
        : dr(_dr), rcutsq(_rcutsq), qi(_qi), qj(_qj), epsilon(_params.epsilon),
          lperp(_params.lperp), lpar(_params.lpar)
        {
        }

    //! Whether the pair potential uses shape.
    HOSTDEVICE static bool needsShape()
        {
        return false;
        }

    //! Whether the pair potential needs particle tags.
    HOSTDEVICE static bool needsTags()
        {
        return false;
        }

    //! whether pair potential requires charges
    HOSTDEVICE static bool needsCharge()
        {
        return false;
        }

    /// Whether the potential implements the energy_shift parameter
    HOSTDEVICE static bool constexpr implementsEnergyShift()
        {
        return true;
        }

    //! Accept the optional shape values
    /*! \param shape_i Shape of particle i
        \param shape_j Shape of particle j
    */
    HOSTDEVICE void setShape(const shape_type* shapei, const shape_type* shapej) { }

    //! Accept the optional tags
    /*! \param tag_i Tag of particle i
        \param tag_j Tag of particle j
    */
    HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj) { }

    //! Accept the optional charge values
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    HOSTDEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy
    /*! \param force Output parameter to write the computed force.
        \param pair_eng Output parameter to write the computed pair energy.
        \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the
       cutoff. \param torque_i The torque exerted on the i^th particle. \param torque_j The torque
       exerted on the j^th particle. \return True if they are evaluated or false if they are not
       because we are beyond the cutoff.
    */
    HOSTDEVICE bool evaluate(Scalar3& force,
                             Scalar& pair_eng,
                             bool energy_shift,
                             Scalar3& torque_i,
                             Scalar3& torque_j)
        {

        Qtwo::Quaternion qA(qi.s,qi.v.x,qi.v.y,qi.v.z);
        Qtwo::Quaternion qB(qj.s,qj.v.x,qj.v.y,qj.v.z);
        const double vr[3] = {dr.x,dr.y,dr.z};
//std::cout<<"r="<<dr.x<<","<<dr.y<<","<<dr.z<<std::endl;
std::cout<<"qa="<<qi.s<<","<<qi.v.x<<","<<qi.v.y<<","<<qi.v.z<<std::endl;
std::cout<<"qb="<<qj.s<<","<<qj.v.x<<","<<qj.v.y<<","<<qj.v.z<<std::endl;
std::cout<<"vr="<<vr[0]<<","<<vr[1]<<","<<vr[2]<<std::endl;
        Qtwo::PotentialPrism pot(qA, qB, vr);
pot.print(std::cout);
        pair_eng = pot.evaluate();
std::cout<<"Energy = "<<pair_eng<<std::endl;
        // compute torque on particle A
        double tau0, tau1, tau2, tau3;
        pot.torqueA(tau0,tau1,tau2,tau3);
        const double tau0A = tau0;
std::cout<<"Tau A = "<<tau1<<","<<tau2<<","<<tau3<<std::endl;

        torque_i = make_scalar3(-0.5*tau1,-0.5*tau2,-0.5*tau3);
//std::cout<<"tau="<<tau0<<","<<tau1<<","<<tau2<<","<<tau3<<std::endl;

        // compute torque on particle B
        pot.torqueB(tau0,tau1,tau2,tau3);
        const double tau0B = tau0;
std::cout<<"Tau B = "<<tau1<<","<<tau2<<","<<tau3<<std::endl;

        torque_j = make_scalar3(-0.5*tau1,-0.5*tau2,-0.5*tau3);
//std::cout<<"tau="<<tau0B<<","<<tau1<<","<<tau2<<","<<tau3<<std::endl;

        // verify first component is 0
        if(std::abs(tau0A+tau0B)>1.e-6)
        {
           std::cout<<"tau0A = "<<tau0A<<std::endl;
           std::cout<<"tau0B = "<<tau0B<<std::endl;
           std::cout<<"Expected tau0 = 0."<<std::endl;
           abort();
        }
        // compute vector force
        const double fr = -1.*pot.devaldr();
        const double vrnomi = 1./std::sqrt(vr[0]*vr[0]+vr[1]*vr[1]+vr[2]*vr[2]);
        force.x = fr*vr[0]*vrnomi;
        force.y = fr*vr[1]*vrnomi;
        force.z = fr*vr[2]*vrnomi;
//std::cout<<"f="<<force.x<<","<<force.y<<","<<force.z<<std::endl;
        return true;
        }

    DEVICE Scalar evalPressureLRCIntegral()
        {
        return 0;
        }

    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return 0;
        }

#ifndef __HIPCC__
    //! Get the name of the potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return "gb";
        }

    std::string getShapeSpec() const
        {
        std::ostringstream shapedef;
        shapedef << "{\"type\": \"Ellipsoid\", \"a\": " << lperp << ", \"b\": " << lperp
                 << ", \"c\": " << lpar << "}";
        return shapedef.str();
        }
#endif

    protected:
    vec3<Scalar> dr; //!< Stored dr from the constructor
    Scalar rcutsq;   //!< Stored rcutsq from the constructor
    quat<Scalar> qi; //!< Orientation quaternion for particle i
    quat<Scalar> qj; //!< Orientation quaternion for particle j
    Scalar epsilon;
    Scalar lperp;
    Scalar lpar;
    // const param_type &params;  //!< The pair potential parameters
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __EVALUATOR_PAIR_TWOQ_H__
