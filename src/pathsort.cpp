#include <stdio.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <algorithm>
#include "src/pathsort.h"
#include "avx2sort.h"
#define __SIMD_BITONIC_IMPLEMENTATION__
#include "simd_bitonic.h"

//---
#define ASCENDING(a, b, c, d, e, f, g, h)                                    \
  (((h < 7) << 7) | ((g < 6) << 6) | ((f < 5) << 5) | ((e < 4) << 4) | \
      ((d < 3) << 3) | ((c < 2) << 2) | ((b < 1) << 1) | (a < 0))

//---
#define DESCENDING(a, b, c, d, e, f, g, h)                                    \
  (((a < 0) << 7) | ((b < 1) << 6) | ((c < 2) << 5) | ((d < 3) << 4) | \
      ((e < 4) << 3) | ((f < 5) << 2) | ((g < 6) << 1) | (h < 7))

//---
#define SORT8(V, ASC) \
{ \
  constexpr int shuffle_masks[3] = { _MM_SHUFFLE(2, 3, 0, 1), \
                                     _MM_SHUFFLE(1, 0, 3, 2), \
                                     _MM_SHUFFLE(3, 1, 2, 0) }; \
  constexpr int blend_masks[8] = { DESCENDING(1, 0, 3, 2, 5, 4, 7, 6), \
                                   DESCENDING(2, 3, 0, 1, 6, 7, 4, 5), \
                                   DESCENDING(0, 2, 1, 3, 4, 6, 5, 7), \
                                   DESCENDING(7, 6, 5, 4, 3, 2, 1, 0), \
                                   ASCENDING(1, 0, 3, 2, 5, 4, 7, 6), \
                                   ASCENDING(2, 3, 0, 1, 6, 7, 4, 5), \
                                   ASCENDING(0, 2, 1, 3, 4, 6, 5, 7), \
                                   ASCENDING(7, 6, 5, 4, 3, 2, 1, 0) }; \
  __m256i shuffled = _mm256_shuffle_epi32(V, shuffle_masks[0]); \
  __m256i min = _mm256_min_epi32(shuffled, V); \
  __m256i max = _mm256_max_epi32(shuffled, V); \
  V = _mm256_blend_epi32(min, max, blend_masks[ASC << 2]); \
  shuffled = _mm256_shuffle_epi32(V, shuffle_masks[1]); \
  min = _mm256_min_epi32(shuffled, V);  \
  max = _mm256_max_epi32(shuffled, V); \
  V = _mm256_blend_epi32(min, max, blend_masks[(ASC << 2) + 1]); \
  shuffled = _mm256_shuffle_epi32(V, shuffle_masks[2]); \
  min = _mm256_min_epi32(shuffled, V); \
  max = _mm256_max_epi32(shuffled, V); \
  V = _mm256_blend_epi32(min, max, blend_masks[(ASC << 2) + 2]); \
  __m256i reverse = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0); \
  __m256i reversed = _mm256_permutevar8x32_epi32(V, reverse); \
  min = _mm256_min_epi32(reversed, V); \
  max = _mm256_max_epi32(reversed, V); \
  V = _mm256_blend_epi32(min, max, blend_masks[(ASC << 2) + 3]); \
  shuffled = _mm256_shuffle_epi32(V, shuffle_masks[1]); \
  min = _mm256_min_epi32(shuffled, V);  \
  max = _mm256_max_epi32(shuffled, V); \
  V = _mm256_blend_epi32(min, max, blend_masks[(ASC << 2) + 1]); \
  shuffled = _mm256_shuffle_epi32(V, shuffle_masks[0]); \
  min = _mm256_min_epi32(shuffled, V);  \
  max = _mm256_max_epi32(shuffled, V); \
  V = _mm256_blend_epi32(min, max, blend_masks[ASC << 2]); \
}

//---
#define MERGE16_ASCENDING(V0, V1) \
{ \
  __m256i _min = _mm256_min_epi32(V0, V1); \
  int ordered = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(V0, _min))); \
  if (ordered == 0x00) {\
    __m256i t = V0; \
    V0 = V1; \
    V1 = t; \
  } else if (ordered < 0xFF) { \
    __m256i _max = _mm256_max_epi32(V0, V1); \
    V0 = _min; \
    V1 = _max; \
    SORT8(V0, true); \
    SORT8(V1, true); \
  } \
}

//---
#define MERGE16_DESCENDING(V0, V1) \
{ \
  __m256i _min = _mm256_min_epi32(V0, V1); \
  int ordered = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(V0, _min))); \
  if (ordered == 0xFF) {\
    __m256i t = V0; \
    V0 = V1; \
    V1 = t; \
  } else if (ordered > 0x00) { \
    __m256i _max = _mm256_max_epi32(V0, V1); \
    V0 = _max; \
    V1 = _min; \
    SORT8(V0, false); \
    SORT8(V1, false); \
  } \
}

//---
void PathSort::sort_bitonic(int* keys,
                    void* values,
                    unsigned int count)
{
  __m256i* _keys = (__m256i*)keys;
  unsigned int level_counts[32] = { 0 };
  unsigned int total_registers = count >> 3;
  for (unsigned int r = 0; r < total_registers; r += 2) {
    __m256i r0 = _mm256_load_si256(&_keys[r]);
    __m256i r1 = _mm256_load_si256(&_keys[r + 1]);
    SORT8(r0, true);
    SORT8(r1, false);
    if (r & 0x2) {
      MERGE16_DESCENDING(r0, r1);
    } else {
      MERGE16_ASCENDING(r0, r1);
    }
    _mm256_store_si256(&_keys[r], r0);
    _mm256_store_si256(&_keys[r + 1], r1);
  }
  for (unsigned int r = 0; r < total_registers; r += 4) {
    unsigned int level = 1;
    do {
      ++level;
      const unsigned int batch_registers = 0x1 << (level - 1);
      const bool ascending = ((r & (0x1 << level)) == 0);
      __m256i* left = &_keys[(r >> level) << level];
      __m256i* right = left + batch_registers;
      const unsigned int registers = 0x1 << level;
      for (unsigned int descent = level; descent > 0; --descent) {
        const unsigned int splits = registers >> descent;
        const unsigned int split_registers = 0x1 << (descent - 1);
        for (unsigned int s = 0; s < splits; ++s) {
          __m256i* nleft = &left[s << descent];
          __m256i* nright = nleft + split_registers;
          for (unsigned int n = 0; n < split_registers; ++n) {
            __m256i L = _mm256_load_si256(nleft);
            __m256i R = _mm256_load_si256(nright);
            __m256i _min = _mm256_min_epi32(L, R);
            __m256i _max = _mm256_max_epi32(L, R);
            _mm256_store_si256(nleft, ascending ? _min : _max);
            _mm256_store_si256(nright, ascending ? _max : _min);
            ++nleft;
            ++nright;
          }
        }
      }
      for (unsigned int f = 0; f < registers; f += 2) {
        __m256i L = _mm256_load_si256(left + f);
        __m256i R = _mm256_load_si256(left + f + 1);
        if (ascending) {
          SORT8(L, true);
          SORT8(R, true);
        } else {
          SORT8(L, false);
          SORT8(R, false);
        }
        _mm256_store_si256(left + f, L);
        _mm256_store_si256(left + f + 1, R);
      }
    } while (!(++level_counts[level] & 0x1));
  }
}

//---
unsigned long long ticks_now()
{
  LARGE_INTEGER now_ticks;
  QueryPerformanceCounter(&now_ticks);
  return now_ticks.QuadPart;
}

//---
int main()
{
  Random random(ticks_now());
  const int count = 65536;
  int* keys = (int*)_aligned_malloc(sizeof(int) * count, 32);
  PathSort pathsort;


  for (int i = 0; i < 10; ++i) {
    for (int i = 0; i < count; ++i) {
      keys[i] = random.next() & 0xFFFFFFFF;
 //     printf("%u\n", values[i]);
    }
    unsigned long long now = ticks_now();
    pathsort.sort_bitonic(keys, nullptr, count);
    //std::sort(keys, keys + count);
    //simd_merge_sort((float*)keys, count);
    //avx2::quicksort(keys, count);
    printf("%llu ticks\n", ticks_now() - now);

    
    for (unsigned int i = 0; i < count - 1; ++i) {
      //    printf("%i\n", keys[i]);
      if (keys[i] > keys[i + 1]) {
        printf("FUCK\n");
        printf("%i vs %i\n", keys[i], keys[i+1]);
      }
    }
  }

  _aligned_free(keys);
  return 0;
}