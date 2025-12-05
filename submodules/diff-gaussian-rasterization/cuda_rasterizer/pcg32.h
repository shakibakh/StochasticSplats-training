
/*
Adopted from https://github.com/wjakob/pcg32
*/
#ifndef CUDA_RASTERIZER_PCG32_H_INCLUDED
#define CUDA_RASTERIZER_PCG32_H_INCLUDED

#define PCG32_MULT           0x5851f42d4c957f2dULL

#include <cuda.h>

struct pcg32_state
{
    uint64_t state;
    uint64_t inc;
};

__forceinline__ __device__ uint32_t pcg32_rand(pcg32_state* rng_state)
{
    uint64_t oldstate = rng_state->state;
    rng_state->state = oldstate * PCG32_MULT + rng_state->inc;
    uint32_t xorshifted = (uint32_t) (((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t) (oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
}

__forceinline__ __device__ float pcg32_float(pcg32_state* rng_state)
{
    union {
        uint32_t u;
        float f;
    } x;
    x.u = (pcg32_rand(rng_state) >> 9) | 0x3f800000u;
    return x.f - 1.0f;
}

__forceinline__ __device__ float2 pcg32_float2(pcg32_state* rng_state)
{
    float2 random_floats;
    random_floats.x = pcg32_float(rng_state);
    random_floats.y = pcg32_float(rng_state);
    return random_floats;
}

__forceinline__ __device__ void pcg32_srandom(pcg32_state* rng_state, uint64_t initstate, uint64_t initseq)
{
    rng_state->state = 0u;
    rng_state->inc = (initseq << 1u) | 1u;
    pcg32_rand(rng_state);
    rng_state->state += initstate;
    pcg32_rand(rng_state);
}

#endif