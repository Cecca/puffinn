#pragma once

#include "typedefs.hpp"

namespace puffinn {

//! Stores the hash values of all points in order 
//! to detect duplicate collisions
class Deduplicator {
    //! The number of repetitions
    size_t num_repetitions;
    //! The hashes for each input point, for each repetition.
    //! This is organized as a matrix in row major order: on the rows
    //! we have the points, on the columns we have the repetitions.
    std::vector<LshDatatype> hashes;

    public:

    Deduplicator(): num_repetitions(0) {}
    Deduplicator(size_t num_repetitions): num_repetitions(num_repetitions) {}

    bool is_empty() const {
        return hashes.size() == 0;
    }

    void resize(size_t n) {
        hashes.resize(n * num_repetitions);
    }

    void insert(size_t i, size_t repetition, LshDatatype h) {
        size_t offset = i * num_repetitions;
        hashes[offset + repetition] = h;
    }

    //! Return the index of the first repetition in which the two given
    //! points collided, or -1 if they never collide. This is useful to 
    //! compute the similarity of points only in the first repetition.
    size_t first_collision_at(size_t i, size_t j, size_t prefix) const {
        LshDatatype prefix_mask = 0xffffffff << (MAX_HASHBITS - prefix);
        auto ptr_i = hashes.cbegin() + num_repetitions * i;
        auto ptr_j = hashes.cbegin() + num_repetitions * j;
        for (size_t rep=0; rep<num_repetitions; rep++) {
            if ((*ptr_i & prefix_mask) == (*ptr_j & prefix_mask)) {
                return rep;
            }
            ptr_i++;
            ptr_j++;
        }
        return -1;
    }

    /// The memory usage of a single repetition
    static uint64_t repetition_memory_usage(size_t n) {
        return n * sizeof(LshDatatype);
    }
};

}
