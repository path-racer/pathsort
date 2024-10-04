#pragma once
#include <immintrin.h>

//---
struct PathSort
{
  //---
  unsigned int _permute_table[256];
  char         _permute_table_small[768];
  __m256i      _permute_table_avx[256];

  //---
  PathSort() {}
  ~PathSort() {}
  void merge_asc_asc(int* left,
                     int* right,
                     unsigned int count);
  void merge_asc_desc(int* left,
                      int* right,
                      unsigned int count);
  void merge_desc_asc(int* left,
                      int* right,
                      unsigned int count);
  void merge_desc_desc(int* left,
                       int* right,
                       unsigned int count);
  void sort_bitonic(int* keys,
                    void* values,
                    unsigned int count);
};

//---
// Fast pseudo-random number generator Xoshiro256.
struct Random
{
  //---
  unsigned long long shuffles[4];

  //---
  Random(unsigned long long seed)
  {
    for (unsigned int s = 0; s < 8; ++s) {
      seed += 0x9E3779B97F4A7C15LLU;
    }
    for (unsigned int s = 0; s < 4; ++s) {
      unsigned long long z = (seed += 0x9E3779B97F4A7C15LLU);
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9LLU;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBLLU;
      shuffles[s] = z ^ (z >> 31);
    }
  }

  //---
  ~Random() {}

  //---
  unsigned long long next()
  {
    const unsigned long long result = shuffles[0] + shuffles[3];
    const unsigned long long t = shuffles[1] << 17;
    shuffles[2] ^= shuffles[0];
    shuffles[3] ^= shuffles[1];
    shuffles[1] ^= shuffles[2];
    shuffles[0] ^= shuffles[3];
    shuffles[2] ^= t;
    shuffles[3] = (shuffles[3] << 45) | (shuffles[3] >> (64 - 45));
    return result;
  }
};