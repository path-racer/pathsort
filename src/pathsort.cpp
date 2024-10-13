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
#define SORT8_ALREADY_BITONIC_ASC(A){\
  __m256i p = _mm256_permute2x128_si256(A, A, 0x1); \
  __m256i _min = _mm256_min_epi32(A, p); \
  __m256i _max = _mm256_max_epi32(A, p); \
  A = _mm256_blend_epi32(_min, _max, 0xF0); \
  p = _mm256_shuffle_epi32(A, _MM_SHUFFLE(1, 0, 3, 2)); \
  _min = _mm256_min_epi32(A, p); \
  _max = _mm256_max_epi32(A, p); \
  A = _mm256_blend_epi32(_min, _max, 0xCC); \
  p = _mm256_shuffle_epi32(A, _MM_SHUFFLE(2, 3, 0, 1)); \
  _min = _mm256_min_epi32(A, p); \
  _max = _mm256_max_epi32(A, p); \
  A = _mm256_blend_epi32(_min, _max, 0xAA);}

//---
#define SORT8_ALREADY_BITONIC_DESC(A){\
  __m256i p = _mm256_permute2x128_si256(A, A, 0x1); \
  __m256i _min = _mm256_min_epi32(A, p); \
  __m256i _max = _mm256_max_epi32(A, p); \
  A = _mm256_blend_epi32(_max, _min, 0xF0); \
  p = _mm256_shuffle_epi32(A, _MM_SHUFFLE(1, 0, 3, 2)); \
  _min = _mm256_min_epi32(A, p); \
  _max = _mm256_max_epi32(A, p); \
  A = _mm256_blend_epi32(_max, _min, 0xCC); \
  p = _mm256_shuffle_epi32(A, _MM_SHUFFLE(2, 3, 0, 1)); \
  _min = _mm256_min_epi32(A, p); \
  _max = _mm256_max_epi32(A, p); \
  A = _mm256_blend_epi32(_max, _min, 0xAA);}

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
  V1 = _mm256_max_epi32(V0, V1); \
  V0 = _min; \
  SORT8_ALREADY_BITONIC_ASC(V0); \
  SORT8_ALREADY_BITONIC_ASC(V1); \
}

//---
#define MERGE16_DESCENDING(V0, V1) \
{ \
  __m256i _max = _mm256_max_epi32(V0, V1); \
  V1 = _mm256_min_epi32(V0, V1); \
  V0 = _max; \
  SORT8_ALREADY_BITONIC_DESC(V0); \
  SORT8_ALREADY_BITONIC_DESC(V1); \
}

//---
#define BINARY_SEARCH_ASC(BITONIC_POINT, COMPARATOR) \
{ \
  int low = search_start; \
  int high = search_end; \
  int comparison = 1; \
  while (low <= high) { \
    BITONIC_POINT = low + ((high - low) >> 1); \
    int L = left[BITONIC_POINT]; \
    int R = right[BITONIC_POINT - search_start]; \
    comparison = COMPARATOR; \
    high -= comparison * (high - BITONIC_POINT + 1); \
    low += !comparison * (BITONIC_POINT - low + 1); \
  } \
  BITONIC_POINT += !comparison; \
}

//---
#define BINARY_SEARCH_DESC(BITONIC_POINT, COMPARATOR) \
{ \
  int low = search_start; \
  int high = search_end; \
  int comparison = 1; \
  while (low <= high) { \
    BITONIC_POINT = low + ((high - low) >> 1); \
    int L = left[BITONIC_POINT - search_start]; \
    int R = right[BITONIC_POINT]; \
    comparison = COMPARATOR; \
    high -= comparison * (high - BITONIC_POINT + 1); \
    low += !comparison * (BITONIC_POINT - low + 1); \
  } \
  BITONIC_POINT += !comparison; \
}

//---
#define SWAPS(SWAP_LEFT, SWAP_RIGHT) \
{ \
  for (unsigned int n = SWAP_LEFT; n < SWAP_RIGHT; ++n) { \
    int t = left[n]; \
    left[n] = right[n - search_start]; \
    right[n - search_start] = t; \
  } \
}

//---
#define RECURSE(F0, F1) \
{ \
  if (bitonic_point < search_end) { \
    if (bitonic_point > 1) { \
      F0(left, left + bitonic_point, left + left_count); \
    } \
    if (bitonic_point > search_start) { \
      F1(right, right + bitonic_point - search_start, right + right_count); \
    } \
  } \
}

//---
void PathSort::merge_asc_asc(int* left,
                             int* right,
                             int* end)
{
  unsigned int left_count = right - left;
  unsigned int right_count = end - right;
  unsigned int search_start = (left_count <= right_count) ? 0 : (left_count - right_count);
  unsigned int search_end = left_count;
  // Find the bitonic point between the two arrays.
  unsigned int bitonic_point = 0;
  BINARY_SEARCH_ASC(bitonic_point, (L > R));
  // Swap inclusively after the bitonic point.
  SWAPS(bitonic_point, search_end);
  // Recurse on the two new bitonic sequences based on the resulting shapes.
  RECURSE(merge_asc_asc, merge_asc_desc);
}

//---
// Here we alter the search to account for local descension, and swap before the bitonic point
// to maintain global ascension.
void PathSort::merge_asc_desc(int* left,
                              int* right,
                              int* end)
{
  unsigned int left_count = right - left;
  unsigned int right_count = end - right;
  unsigned int search_start = (right_count <= left_count) ? 0 : (right_count - left_count);
  unsigned int search_end = right_count;
  unsigned int bitonic_point = 0;
  BINARY_SEARCH_DESC(bitonic_point, (L < R));
  // Swap exclusively before the bitonic point.
  SWAPS(search_start, bitonic_point);
  RECURSE(merge_asc_asc, merge_asc_desc);
}

//---
void PathSort::merge_desc_asc(int* left,
                              int* right,
                              int* end)
{
  unsigned int left_count = right - left;
  unsigned int right_count = end - right;
  unsigned int search_start = (left_count <= right_count) ? 0 : (left_count - right_count);
  unsigned int search_end = left_count;
  unsigned int bitonic_point = 0;
  BINARY_SEARCH_ASC(bitonic_point, (L > R));
  // Swap inclusively before the bitonic point.
  SWAPS(search_start, bitonic_point + 1);
  RECURSE(merge_desc_desc, merge_desc_asc);
}

//---
void PathSort::merge_desc_desc(int* left,
                               int* right,
                               int* end)
{
  unsigned int left_count = right - left;
  unsigned int right_count = end - right;
  unsigned int search_start = (right_count <= left_count) ? 0 : (right_count - left_count);
  unsigned int search_end = right_count;
  unsigned int bitonic_point = 0;
  BINARY_SEARCH_DESC(bitonic_point, (L < R));
  // Swap exclusively after the bitonic point.
  SWAPS(bitonic_point, search_end);
  RECURSE(merge_desc_desc, merge_desc_asc);
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
    __m256i* r0 = &_keys[r];
    __m256i* r1 = &_keys[r + 1];
    SORT8(*r0, true);
    SORT8(*r1, false);
    if (r & 0x2) {
      MERGE16_DESCENDING(*r0, *r1);
    } else {
      MERGE16_ASCENDING(*r0, *r1);
    }
  }

  printf("--------\n");
  for (unsigned int i = 0; i < count; ++i) {
    printf("%i\n", keys[i]);
  }
  printf("--------\n");

  for (unsigned int r = 0; r < total_registers; r += 4) {
    unsigned int level = 1;
    do {
      ++level;
      const unsigned int batch_elements = 0x8 << (level - 1);
      int ascending = (r & (0x1 << level)) == 0;
      int* left = (int*)&_keys[(r >> level) << level];
      int* right = (int*)(left + batch_elements);
      if (ascending) {
        merge_asc_asc(left, right, right + batch_elements);
      } else {
        merge_desc_asc(left, right, right + batch_elements);
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
  /*
  __m256i V0 = _mm256_setr_epi32(9, 9, 9, 9, 9, 9, 8, 8);
  __m256i V1 = _mm256_setr_epi32(9, 9, 9, 8, 8, 8, 8, 8);
  __m256i _min = _mm256_min_epi32(V0, V1);
  int ordered = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(V0, _min)));
  if (ordered == 0x00) {
      __m256i t = V0;
      V0 = V1;
      V1 = t;
  }
  else if (ordered < 0xFF) {
      __m256i _max = _mm256_max_epi32(V0, V1);
      V0 = _min;
      V1 = _max;
      SORT8(V0, true);
      SORT8(V1, true);
  }*/

  Random random(ticks_now());
  const int count = 32;
  int* keys = (int*)_aligned_malloc(sizeof(int) * count, 32);
  PathSort pathsort;

  int v[32] =
  {
    0,1,1,1,2,2,2,2, 3,4,4,5,5,6,6,7,
    7,7,7,6,6,5,4,3, 3,3,3,2,1,1,1,0
  };

  for (int i = 0; i < 1; ++i) {
    for (int i = 0; i < count; ++i) {
      keys[i] = random.next() & 0x7;
     // printf("%u\n", keys[i]);
    }

    unsigned long long now = ticks_now();
    pathsort.sort_bitonic(v, nullptr, count);
    //std::sort(keys, keys + count);
    //simd_merge_sort((float*)keys, count);
    //avx2::quicksort(keys, count);
    printf("%llu ticks\n", ticks_now() - now);

    
    for (unsigned int i = 0; i < count - 1; ++i) {
      printf("%i\n", v[i]);
      if (v[i] > v[i + 1]) {
        printf("FUCK\n");
      }
    }
  }

  _aligned_free(keys);
  return 0;
}