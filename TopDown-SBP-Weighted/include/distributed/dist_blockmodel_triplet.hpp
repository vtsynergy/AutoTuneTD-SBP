/**
 * Stores the distributed triplet of blockmodels needed for the fibonacci search.
 */
#ifndef SBP_DIST_BLOCKMODEL_TRIPLET_HPP
#define SBP_DIST_BLOCKMODEL_TRIPLET_HPP

#include "distributed/two_hop_blockmodel.hpp"

class DistBlockmodelTriplet {

public:
    DistBlockmodelTriplet() : optimal_num_blocks_found(false) {}
    /// True if the optimal number of blocks has been found.
    bool optimal_num_blocks_found;
    /// Returns a reference to the blockmodel at position `i`. 0 <= i <= 2. i = 0 has the larger number of blocks.
    TwoHopBlockmodel &get(long i) { return this->blockmodels[i]; }
    /// Selects a blockmodel on which to perform the next SBP iteration. Always selects the "middle" blockmodel until
    /// the golden ratio has been reached.
    virtual TwoHopBlockmodel get_next_blockmodel(TwoHopBlockmodel &old_blockmodel);
    /// Returns false if the golden ratio has been reached (there is a non-empty blockmodel in all 3 spots in the
    /// triplet).
    bool golden_ratio_not_reached();
    /// Returns true if the optimal number of blocks has been found.
    virtual bool is_done();
    /// Updates the triplet with a new blockmodel, based on the blockmodel's entropy and number of blocks.
    virtual void update(TwoHopBlockmodel &blockmodel);
    /// Prints the status of the triplet.
    void status();

protected:
    /// Blockmodels arranged in order of decreasing number of blocks.
    /// If the first blockmodel is empty, then the golden ratio bracket has not yet been established.
    TwoHopBlockmodel blockmodels[3];
    /// Difference in number of blocks between the two blockmodels with the smallest number of blocks.
    virtual long lower_difference();
    /// Difference in number of blocks between the two blockmodels with the highest number of blocks.
    virtual long upper_difference();
};

class DistTopDownBlockmodelTriplet : public DistBlockmodelTriplet {
 public:
    /// Selects a blockmodel on which to perform the next SBP iteration. Always selects the "middle" blockmodel until
    /// the golden ratio has been reached.
    TwoHopBlockmodel get_next_blockmodel(TwoHopBlockmodel &old_blockmodel) override;
    /// Returns true if the optimal number of blocks has been found.
    bool is_done() override;
    /// Updates the triplet with a new blockmodel, based on the blockmodel's entropy and number of blocks.
    void update(TwoHopBlockmodel &blockmodel) override;

protected:
    /// Difference in number of blocks between the two blockmodels with the smallest number of blocks.
    long lower_difference() override;
    /// Difference in number of blocks between the two blockmodels with the highest number of blocks.
    long upper_difference() override;
};

#endif  // SBP_DIST_BLOCKMODEL_TRIPLET_HPP