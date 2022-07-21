#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/similarity_measure/cosine.hpp"
#include "puffinn/hash_source/independent.hpp"
#include "puffinn/hash/simhash.hpp"
#include "puffinn/typedefs.hpp"
#include "puffinn/filterer.hpp"

#include <vector>

namespace puffinn {

static const int DEFAULT_SKETCH_BITS = 64;

template<int Bits>
class Sketch {
private:
    static const int WordBits = 64;
    static const int Words = Bits / WordBits;
    uint64_t bits[Words];

    size_t hamming_distance(const Sketch<Bits> & other) const {
        size_t hd = 0;
        for (size_t w_idx=0; w_idx<Words; w_idx++) {
            hd += popcountll(bits[w_idx] ^ other.bits[w_idx]);
        }
        return hd;
    }

public:

    template<typename T>
    static Sketch<Bits> build_from(const puffinn::Filterer<T> & filterer, size_t idx) {
        Sketch<Bits> s;

        size_t word_idx = 0;
        for (size_t sketch_idx=0; sketch_idx<puffinn::NUM_SKETCHES; sketch_idx++) {
            if (word_idx >= Words) { break; }
            s.bits[word_idx++] = filterer.get_sketch(idx, sketch_idx);
        }

        return s;
    }

    //! Return an estimated upper bound to the collision probability of the two sketches, 
    //! that can be used to derive an upper bound to the similarity of the 
    //! corresponding vectors
    float collision_probability_upper_bound(const Sketch<Bits> & other, float delta) const {
        // probability of bits being different
        float theta = std::sqrt(
            2.0 / Bits * std::log(1.0/delta)
        );
        float est = (((float)Bits) - hamming_distance(other)) / Bits;
        float upper_prob = est + theta;
        if (upper_prob > 1.0) {
            upper_prob = 1.0;
        }
        return upper_prob;
    }
};
}

