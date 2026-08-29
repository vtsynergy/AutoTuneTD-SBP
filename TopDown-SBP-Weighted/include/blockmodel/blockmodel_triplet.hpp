/**
 * Stores the triplet of blockmodels needed for the fibonacci search.
 */
#ifndef CPPSBP_BLOCKMODEL_TRIPLET_HPP
#define CPPSBP_BLOCKMODEL_TRIPLET_HPP

#include <iostream>
#include <limits>

#include "blockmodel.hpp"

class BlockmodelTriplet {

public:
    /// Creates an empty blockmodel triplet.
    BlockmodelTriplet() : optimal_num_blocks_found(false) {}
    /// True if the optimal number of blocks has been found.
    bool optimal_num_blocks_found;
    /// Updates the blockmodel triplet with the provided blockmodel (inserts it into the correct spot, moves/deletes
    /// the remaining blockmodels as needed).
    virtual void update(Blockmodel &blockmodel);
    /// prints the number of blocks and overall entropy of every blockmodel in the blockmodel triplet.
    void status();
    /// Returns the blockmodel at index `i`. IndexOutOfBoundError if i >= 3.
    Blockmodel &get(long i) { return blockmodels[i]; }
    /// Returns true if the golden ratio has not been reached, false otherwise. In practice, returns true if
    /// blockmodels[2] == 0.
    bool golden_ratio_not_reached();
    /// Returns true if the optimal number of blocks has been found.
    virtual bool is_done();
    /// Returns the blockmodel on which to perform the next iteration of SBP.
    virtual Blockmodel get_next_blockmodel(Blockmodel &old_blockmodel);

protected:
    /// Blockmodels arranged in order of decreasing number of blocks.
    /// If the first blockmodel is empty, then the golden ratio bracket has not yet been established.
    Blockmodel blockmodels[3];
    /// TODO
    virtual long lower_difference();
    /// TODO
    virtual long upper_difference();
};

class TopDownBlockmodelTriplet : public BlockmodelTriplet {
public:
    void update(Blockmodel &blockmodel) override;
    bool is_done() override;
    Blockmodel get_next_blockmodel(Blockmodel &old_blockmodel) override;

protected:
    long lower_difference() override;
    long upper_difference() override;
};

#endif // CPPSBP_BLOCKMODEL_TRIPLET_HPP
