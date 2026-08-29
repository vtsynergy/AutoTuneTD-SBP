#include "blockmodel_triplet.hpp"
#include "mpi_data.hpp"

Blockmodel BlockmodelTriplet::get_next_blockmodel(Blockmodel &old_blockmodel) {
    old_blockmodel.setNum_blocks_to_merge(0);
    this->update(old_blockmodel);
    this->status();

    // If search has not yet reached golden ratio, continue from middle blockmodel
    if (this->golden_ratio_not_reached()) {
        Blockmodel blockmodel = this->get(1).copy();
        blockmodel.setNum_blocks_to_merge(long(blockmodel.num_blocks() * BLOCK_REDUCTION_RATE));
        if (blockmodel.getNum_blocks_to_merge() == 0) {
            this->optimal_num_blocks_found = true;
        }
        return blockmodel;
    }
    // If community detection is done, return the middle (optimal) blockmodel
    if (this->is_done()) {
        return this->get(1).copy();
    }
    // Find which Blockmodel would serve as the starting point for the next iteration
    long index = 1;
    // TODO: if things get funky, look into this if/else statement
    if (this->get(0).empty && this->get(1).num_blocks() > this->get(2).num_blocks()) {
        index = 1;
    } else if (this->upper_difference() >= this->lower_difference()) {
        index = 0;
    } else {
        index = 1;
    }
    long next_num_blocks_to_try = this->get(index + 1).num_blocks();
    next_num_blocks_to_try += long((this->get(index).num_blocks() - this->get(index + 1).num_blocks()) * 0.618);
    Blockmodel blockmodel = this->get(index).copy();
    blockmodel.setNum_blocks_to_merge(blockmodel.num_blocks() - next_num_blocks_to_try);
    return blockmodel;
}

bool BlockmodelTriplet::golden_ratio_not_reached() {
    return this->get(2).empty;
}

bool BlockmodelTriplet::is_done() {
    if ((!this->get(0).empty && this->get(0).num_blocks() - this->get(2).num_blocks() == 2) ||
        (this->get(0).empty && this->get(1).num_blocks() - this->get(2).num_blocks() == 1)) {
        this->optimal_num_blocks_found = true;
    }
    return this->optimal_num_blocks_found;
}

long BlockmodelTriplet::lower_difference() {
    return this->get(1).num_blocks() - this->get(2).num_blocks();
}

long BlockmodelTriplet::upper_difference() {
    return this->get(0).num_blocks() - this->get(1).num_blocks();
}

void BlockmodelTriplet::status() {
    double entropies[3];
    long num_blocks[3];
    for (long i = 0; i < 3; ++i) {
        if (this->blockmodels[i].empty) {
            entropies[i] = std::numeric_limits<double>::min();
            num_blocks[i] = 0;
        } else {
            entropies[i] = this->blockmodels[i].getOverall_entropy();
            num_blocks[i] = this->blockmodels[i].num_blocks();
        }
    }
    if (mpi.rank == 0) {
        std::cout << mpi.rank << " | Overall entropy: " << entropies[0] << " " << entropies[1] << " " << entropies[2] << std::endl << std::flush;
        std::cout << mpi.rank << " | Number of blocks: " << num_blocks[0] << " " << num_blocks[1] << " " << num_blocks[2] << std::endl << std::flush;
        if (this->optimal_num_blocks_found) {
            std::cout << mpi.rank << " | Optimal blockmodel found with " << num_blocks[1] << " blocks" << std::endl << std::flush;
        } else if (!(this->golden_ratio_not_reached())) {
            std::cout << mpi.rank << " | Golden ratio has been reached" << std::endl << std::flush;
        }
    }
}

void BlockmodelTriplet::update(Blockmodel &blockmodel) {
    long index;
    if (this->blockmodels[1].empty) {
        index = 1;
    } else {
        if (blockmodel.getOverall_entropy() <= this->blockmodels[1].getOverall_entropy()) {
            long old_index;
            if (this->get(1).num_blocks() > blockmodel.num_blocks()) {
                old_index = 0;
            } else {
                old_index = 2;
            }
            this->blockmodels[old_index] = this->blockmodels[1].copy();
            index = 1;
        } else {
            if (this->blockmodels[1].num_blocks() > blockmodel.num_blocks()) {
                index = 2;
            } else {
                index = 0;
            }
        }
    }
    this->blockmodels[index] = blockmodel.copy();
}

Blockmodel TopDownBlockmodelTriplet::get_next_blockmodel(Blockmodel &old_blockmodel) {
    old_blockmodel.setNum_blocks_to_merge(0);
    this->update(old_blockmodel);
    this->status();

    // If search has not yet reached golden ratio, continue from middle blockmodel
    if (this->golden_ratio_not_reached()) {
        Blockmodel blockmodel = this->get(1).copy();
//        blockmodel.setNum_blocks_to_merge(long(blockmodel.num_blocks() * (1.0/BLOCK_REDUCTION_RATE)));
        blockmodel.setNum_blocks_to_merge(long(ceil(blockmodel.num_blocks() * args.splitrate)));
        if (blockmodel.getNum_blocks_to_merge() == 0 ||
            blockmodel.getNum_blocks_to_merge() >= (long) blockmodel.block_assignment().size()) {
            this->optimal_num_blocks_found = true;
        }
        return blockmodel;
    }
    std::cout << "Golden ratio has been reached!" << std::endl;
    // If community detection is done, return the middle (optimal) blockmodel
    if (this->is_done()) {
        return this->get(1).copy();
    }
    // Find which Blockmodel would serve as the starting point for the next iteration
    long index = 1;
    // TODO: if things get funky, look into this if/else statement
    if (this->get(0).empty && this->get(1).num_blocks() < this->get(2).num_blocks()) {
        index = 1;
    } else if (this->upper_difference() <= this->lower_difference()) {
        index = 1;
    } else {
        index = 0;
    }
//    std::cout << index + 1 << std::endl;
    long next_num_blocks_to_try = this->get(index + 1).num_blocks();
    next_num_blocks_to_try += long((this->get(index).num_blocks() - this->get(index + 1).num_blocks()) * 0.618);
    Blockmodel blockmodel = this->get(index).copy();
    blockmodel.setNum_blocks_to_merge(next_num_blocks_to_try);  //  - blockmodel.num_blocks());
    return blockmodel;
}

bool TopDownBlockmodelTriplet::is_done() {
    if ((!this->get(0).empty && this->get(2).num_blocks() - this->get(0).num_blocks() <= 2) ||
        (this->get(0).empty && this->get(2).num_blocks() - this->get(1).num_blocks() <= 1)) {
        this->optimal_num_blocks_found = true;
    }
    return this->optimal_num_blocks_found;
}

long TopDownBlockmodelTriplet::lower_difference() {
    return this->get(2).num_blocks() - this->get(1).num_blocks();
}

long TopDownBlockmodelTriplet::upper_difference() {
    return this->get(1).num_blocks() - this->get(0).num_blocks();
}

void TopDownBlockmodelTriplet::update(Blockmodel &blockmodel) {
    std::cout << "Received blockmodel with blocks = " << blockmodel.num_blocks() << std::endl;
    long index;
    if (this->blockmodels[1].empty) {
        index = 1;
        std::cout << "placing new blockmodel in the middle (index 1)" << std::endl;
    } else {
        if ((blockmodel.getOverall_entropy() <= this->blockmodels[1].getOverall_entropy()) &&
           (abs(blockmodel.num_blocks() - this->get(1).num_blocks()) > 0.1 * this->get(1).num_blocks())) {
	    long old_index;
            if (this->get(1).num_blocks() < blockmodel.num_blocks()) {
                old_index = 0;
                std::cout << "moving old blockmodel to the start (index 0)" << std::endl;
            } else {
                old_index = 2;
                std::cout << "moving old blockmodel to the end (index 2)" << std::endl;
            }
            this->blockmodels[old_index] = this->blockmodels[1].copy();
            index = 1;
            std::cout << "placing new blockmodel in the middle (index 1)" << std::endl;
        } else {
            if (this->blockmodels[1].num_blocks() < blockmodel.num_blocks()) {
                index = 2;
                std::cout << "placing new blockmodel in the end (index 2)" << std::endl;
            } else {
                index = 0;
                std::cout << "placing new blockmodel in the start (index 0)" << std::endl;
            }
        }
    }
    this->blockmodels[index] = blockmodel.copy();
}
