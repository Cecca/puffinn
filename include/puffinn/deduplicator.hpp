#pragma once

#include "typedefs.hpp"

#if defined(__AVX2__) || defined(__AVX__)
    #include <immintrin.h>
#endif


namespace puffinn {

//! Stores the hash values of all points in order 
//! to detect duplicate collisions
class Deduplicator {
    /// The number of hash words in a 256-bit SIMD vector
    static const size_t WORDS_PER_VEC = 256 / 32;
    /// The stride with which hashes are stored, so that they are 
    /// 32-byte aligned
    size_t stride;
    //! The number of repetitions
    size_t num_repetitions;
    //! The hashes for each input point, for each repetition.
    //! This is organized as a matrix in row major order: on the rows
    //! we have the points, on the columns we have the repetitions.
    std::vector<LshDatatype> hashes;

    public:

    Deduplicator(): num_repetitions(0), stride(0) {}
    Deduplicator(size_t num_repetitions)
    : num_repetitions(num_repetitions)
    {
        stride = 0;
        while (stride < num_repetitions) {
            stride += WORDS_PER_VEC;
        }
    }

    bool is_empty() const {
        return hashes.size() == 0;
    }

    void resize(size_t n) {
        hashes.resize(n * stride);
    }

    void insert(size_t i, size_t repetition, LshDatatype h) {
        size_t offset = i * stride;
        hashes[offset + repetition] = h;
    }

    int32_t first_collision_from_scalar(size_t R, size_t S, size_t prefix, size_t from) const {
        uint32_t prefix_mask = 0xffffffff << (MAX_HASHBITS - prefix);
        auto offset_i = stride * R;
        auto offset_j = stride * S;
        for (size_t rep=0; rep<num_repetitions; rep++) {
            size_t r = (from + rep) % num_repetitions;
            if ((hashes[offset_i + r] & prefix_mask) == (hashes[offset_j + r] & prefix_mask)) {
                return r;
            }
        }
        return -1;
    }

    #ifdef __AVX2__
    int32_t first_collision_from_avx(size_t R, size_t S, size_t prefix, size_t from) const {
        size_t from_aligned = (from / WORDS_PER_VEC) * WORDS_PER_VEC;
        uint32_t prefix_mask = 0xffffffff << (MAX_HASHBITS - prefix);

        __m256i mask = _mm256_set_epi32(
            prefix_mask,
            prefix_mask,
            prefix_mask,
            prefix_mask,
            prefix_mask,
            prefix_mask,
            prefix_mask,
            prefix_mask
        );

        size_t offset_i = stride * R;
        size_t offset_j = stride * S;

        for (
            size_t r=0;
            r < stride;
            r += WORDS_PER_VEC
        ) {
            size_t rep = (from_aligned + r) % stride;
            __m256i tmpR = _mm256_and_si256(
                // FIXME: the standard vector does not allocate aligned memory
                _mm256_loadu_si256((__m256i*)&hashes[offset_i + rep]),
                mask
            );
            __m256i tmpS = _mm256_and_si256(
                _mm256_loadu_si256((__m256i*)&hashes[offset_j + rep]),
                mask
            );
            __m256i cmpres = _mm256_cmpeq_epi32(tmpR, tmpS);

            if (!_mm256_testz_si256(cmpres, cmpres)) {
                // At least one word has a bit set, therefore extract them and
                // look for the first non zero one
                alignas(32) LshDatatype stored[WORDS_PER_VEC];
                _mm256_store_si256((__m256i*)stored, cmpres);
                for (size_t w=0; w<WORDS_PER_VEC; w++) {
                    if (stored[w] != 0 && rep + w < num_repetitions) {
                        int32_t out_rep = rep + w;
                        return out_rep;
                    }
                }
            }
        }

        return -1;
    }
    #endif



    int32_t first_collision_at_scalar(size_t R, size_t S, size_t prefix) const {
        uint32_t prefix_mask = 0xffffffff << (MAX_HASHBITS - prefix);
        auto ptr_i = hashes.cbegin() + stride * R;
        auto ptr_j = hashes.cbegin() + stride * S;
        for (size_t rep=0; rep<num_repetitions; rep++) {
            if ((*ptr_i & prefix_mask) == (*ptr_j & prefix_mask)) {
                return rep;
            }
            ptr_i++;
            ptr_j++;
        }
        return -1;
    }

    #ifdef __AVX2__
    int32_t first_collision_at_avx(size_t R, size_t S, size_t prefix) const {
        uint32_t prefix_mask = 0xffffffff << (MAX_HASHBITS - prefix);


        __m256i mask = _mm256_set_epi32(
            prefix_mask,
            prefix_mask,
            prefix_mask,
            prefix_mask,
            prefix_mask,
            prefix_mask,
            prefix_mask,
            prefix_mask
        );

        size_t i = stride * R;
        size_t j = stride * S;

        for (
            size_t rep=0;
            rep < stride;
            rep += WORDS_PER_VEC
        ) {
            __m256i tmpR = _mm256_and_si256(
                // FIXME: the standard vector does not allocate aligned memory
                _mm256_loadu_si256((__m256i*)&hashes[i]),
                mask
            );
            __m256i tmpS = _mm256_and_si256(
                _mm256_loadu_si256((__m256i*)&hashes[j]),
                mask
            );
            __m256i cmpres = _mm256_cmpeq_epi32(tmpR, tmpS);

            if (!_mm256_testz_si256(cmpres, cmpres)) {
                // At least one word has a bit set, therefore extract them and
                // look for the first non zero one
                alignas(32) LshDatatype stored[WORDS_PER_VEC];
                _mm256_store_si256((__m256i*)stored, cmpres);
                for (size_t w=0; w<WORDS_PER_VEC; w++) {
                    if (stored[w] != 0) {
                        int32_t out_rep = rep + w;
                        return out_rep;
                    }
                }
            }

            i += WORDS_PER_VEC;
            j += WORDS_PER_VEC;
        }

        return -1;
    }
    #endif

public:
    //! Return the index of the first repetition in which the two given
    //! points collided, or -1 if they never collide. This is useful to 
    //! compute the similarity of points only in the first repetition.
    int32_t first_collision_at(size_t R, size_t S, size_t prefix) const {
        #ifdef __AVX2__
        return first_collision_at_avx(R, S, prefix);
        #else
        return first_collision_at_scalar(R, S, prefix);
        #endif
    }

private:
    size_t all_collisions_at_scalar(size_t R, size_t S, size_t prefix, std::vector<size_t> & out) const {
        assert(out.size() == num_repetitions);
        size_t oidx = 0;
        uint32_t prefix_mask = 0xffffffff << (MAX_HASHBITS - prefix);
        auto ptr_i = hashes.cbegin() + stride * R;
        auto ptr_j = hashes.cbegin() + stride * S;
        for (size_t rep=0; rep<num_repetitions; rep++) {
            if ((*ptr_i & prefix_mask) == (*ptr_j & prefix_mask)) {
                out[oidx++] = rep;
            }
            ptr_i++;
            ptr_j++;
        }
        return oidx;
    }

public:
    int32_t compute_at(size_t R, size_t S, size_t prefix) const {
        #ifdef __AVX2__
        return first_collision_from_avx(R, S, prefix, (R + S) % num_repetitions);
        #else
        return first_collision_from_scalar(R, S, prefix, (R + S) % num_repetitions);
        #endif
    }

    /// The memory usage of a single repetition
    static uint64_t repetition_memory_usage(size_t n) {
        return n * sizeof(LshDatatype);
    }
};

}
