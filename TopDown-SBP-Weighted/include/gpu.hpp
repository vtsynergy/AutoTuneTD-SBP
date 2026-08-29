/**
 * Functions for porting the CPU SBP code to the GPU. Currently only contains functions related to the MCMC phase.
 */
#ifndef SBP_GPU_HPP
#define SBP_GPU_HPP

#include <vector>
#include "blockmodel.hpp"
#include "matrix/csr.hpp"
#include "typedefs.hpp"

/**
 * The CSR format Blockmodel.
 */
class GPUBlockmodel {
  public:
    GPUBlockmodel() = default;
    GPUBlockmodel(const Blockmodel &blockmodel) {

    }
    
};

#endif // SBP_GPU_HPP