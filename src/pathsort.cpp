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
#define BINARY_SEARCH(BITONIC_POINT, COMPARATOR, L_OFFSET, R_OFFSET) \
{ \
  int low = search_start; \
  int high = search_end; \
  while (low <= high) { \
    BITONIC_POINT = low + ((high - low) >> 1); \
    int L = left[BITONIC_POINT - L_OFFSET]; \
    int R = right[BITONIC_POINT - R_OFFSET]; \
    int comparison = COMPARATOR; \
    L = left[BITONIC_POINT - L_OFFSET - 1]; \
    R = right[BITONIC_POINT - R_OFFSET - 1]; \
    if (comparison && !COMPARATOR) { \
      break; \
    } \
    high -= comparison * (high - BITONIC_POINT + 1); \
    low += !comparison * (BITONIC_POINT - low + 1); \
  } \
}

//---
#define SWAPS(SWAP_LEFT, SWAP_RIGHT, L_OFFSET, R_OFFSET) \
{ \
  for (unsigned int n = SWAP_LEFT; n < SWAP_RIGHT; ++n) { \
    int t = left[n - L_OFFSET]; \
    left[n - L_OFFSET] = right[n - R_OFFSET]; \
    right[n - R_OFFSET] = t; \
  } \
}

#define REVERSE(A, C) \
{ \
  for (int e = 0; e < (C  >> 1); ++e) { \
    int t = A[e]; \
    A[e] = A[C - 1 - e]; \
    A[C - 1 - e] = t; \
  } \
}

//---
#define RECURSE_ASC(F0, F1, L_OFFSET, R_OFFSET) \
{ \
  int new_left_left = bitonic_point - L_OFFSET; \
  int new_left_right = left_count - new_left_left; \
  int new_right_left = bitonic_point - R_OFFSET; \
  int new_right_right = right_count - new_right_left; \
  if (new_left_left && new_left_right) { \
    F0(left, left + new_left_left, new_left_left, new_left_right); \
  } else if (new_left_right) { \
    REVERSE(left, left_count); \
  } \
  if (new_right_left && new_right_right) { \
    F1(right, right + new_right_left, new_right_left, new_right_right); \
  } else if (new_right_left) { \
    REVERSE(right, right_count); \
  } \
}

//---
#define RECURSE_DESC(F0, F1) \
{ \
  int new_left_left = bitonic_point; \
  int new_left_right = left_count - new_left_left; \
  int new_right_left = bitonic_point; \
  int new_right_right = right_count - new_right_left; \
  if (new_left_left && new_left_right) { \
    F0(left, left + new_left_left, new_left_left, new_left_right); \
  } else if (new_left_right) { \
    REVERSE(left, left_count); \
  } \
  if (new_right_left && new_right_right) { \
    F1(right, right + new_right_left, new_right_left, new_right_right); \
  } else if (new_right_left) { \
    REVERSE(right, right_count); \
  } \
}

//---
void PathSort::merge_asc_asc(int* left,
                             int* right,
                             int left_count,
                             int right_count)
{
  if (right[right_count - 1] > right[-1]) {
    REVERSE(right, right_count);
    return;
  } else if (left_count + right_count <= 512) {
    avx2::quicksort(left, left_count + right_count);
    return;
  }
  int bitonic_point = 0;
  int search_start, search_end, l_offset, r_offset;
  if (left_count > right_count) {
    search_start = left_count - right_count;
    search_end = left_count;
    l_offset = 0;
    r_offset = search_start;
  } else {
    search_start = right_count - left_count;
    search_end = right_count;
    l_offset = search_start;
    r_offset = 0;
  }
  BINARY_SEARCH(bitonic_point, (L > R), l_offset, r_offset);
  SWAPS(bitonic_point, search_end, l_offset, r_offset);
  RECURSE_ASC(merge_asc_asc, merge_asc_desc, l_offset, r_offset);
}

//---
void PathSort::merge_asc_desc(int* left,
                              int* right,
                              int left_count,
                              int right_count)
{
  if (left[0] < right[0]) {
    REVERSE(left, left_count);
    return;
  } else if (left_count + right_count <= 512) {
    avx2::quicksort(left, left_count + right_count);
    return;
  }
  int bitonic_point = 0;
  int search_start = 0;
  int search_end = (left_count > right_count) ? right_count : left_count;
  BINARY_SEARCH(bitonic_point, (L < R), 0, 0);
  SWAPS(search_start, bitonic_point, 0, 0);
  RECURSE_DESC(merge_asc_asc, merge_asc_desc);
}

//---
void PathSort::merge_desc_asc(int* left,
                              int* right,
                              int left_count,
                              int right_count)
{
  if (left[0] > right[0]) {
    REVERSE(left, left_count);
    return;
  } else if (left_count + right_count <= 512) {
    avx2::quicksort(left, left_count + right_count);
    REVERSE(left, left_count + right_count);
    return;
  }
  int bitonic_point = 0;
  int search_start = 0;
  int search_end = (left_count > right_count) ? right_count : left_count;
  BINARY_SEARCH(bitonic_point, (L > R), 0, 0);
  SWAPS(search_start, bitonic_point, 0, 0);
  RECURSE_DESC(merge_desc_desc, merge_desc_asc);
}

//---
void PathSort::merge_desc_desc(int* left,
                               int* right,
                               int left_count,
                               int right_count)
{
  if (right[-1] > right[right_count - 1]) {
    REVERSE(right, right_count);
    return;
  } else if (left_count + right_count <= 512) {
    avx2::quicksort(left, left_count + right_count);
    REVERSE(left, left_count + right_count);
    return;
  }
  int bitonic_point = 0;
  int search_start, search_end, l_offset, r_offset;
  if (left_count > right_count) {
    search_start = left_count - right_count;
    search_end = left_count;
    l_offset = 0;
    r_offset = search_start;
  } else {
    search_start = right_count - left_count;
    search_end = right_count;
    l_offset = search_start;
    r_offset = 0;
  }
  BINARY_SEARCH(bitonic_point, (L < R), l_offset, r_offset);
  SWAPS(bitonic_point, search_end, l_offset, r_offset);
  RECURSE_ASC(merge_desc_desc, merge_desc_asc, l_offset, r_offset);
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

  /*
  printf("--------\n");
  for (unsigned int i = 0; i < count; ++i) {
    printf("%i\n", keys[i]);
  }
  printf("--------\n");*/

  for (unsigned int r = 0; r < total_registers; r += 4) {
    unsigned int level = 1;
    do {
      ++level;
      const unsigned int batch_elements = 0x8 << (level - 1);
      int ascending = (r & (0x1 << level)) == 0;
      int* left = (int*)&_keys[(r >> level) << level];
      int* right = (int*)(left + batch_elements);
      if (ascending) {
        merge_asc_asc(left, right, batch_elements, batch_elements);
      } else {
        merge_desc_asc(left, right, batch_elements, batch_elements);
      }
    } while (!(++level_counts[level] & 0x1));
  }
}

//---
void PathSort::sort_avx_asc(int* keys,
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

  for (unsigned int r = 0; r < total_registers; r += 4) {
    unsigned int level = 1;
    do {
      ++level;
      const unsigned int batch_registers = 0x1 << (level - 1);
      bool ascending = (r & (0x1 << level)) == 0;
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
      for (unsigned int f = 0; f < registers; ++f) {
        __m256i* L = &left[f];
        if (ascending) {
          SORT8_ALREADY_BITONIC_ASC(*L);
        } else {
          SORT8_ALREADY_BITONIC_DESC(*L);
        }
      }
    } while (!(++level_counts[level] & 0x1));
  }
}

//---
void PathSort::radix_sort(unsigned int* values,
                          unsigned int count,
                          unsigned int shift)
{
  int* new_values = (int*)malloc(sizeof(int) * count);
  unsigned int bucket_offsets[256] = { 0 };
  unsigned int bucket_counter[256] = { 0 };
  for (unsigned int i = 0; i < count; ++i) {
    ++bucket_offsets[(values[i] >> shift) & 0xFF];
  }
  unsigned int previous_count = bucket_offsets[0];
  bucket_offsets[0] = 0;
  for (unsigned int b = 1; b < 256; ++b) {
    unsigned int current = bucket_offsets[b];
    bucket_offsets[b] = bucket_offsets[b - 1] + previous_count;
    previous_count = current;
  }
  for (unsigned int i = 0; i < count; ++i) {
    int v = (values[i] >> shift) & 0xFF;
    new_values[bucket_offsets[v] + (bucket_counter[v]++)] = values[i];
  }
  for (unsigned int b = 0; b < 256; ++b) {
    unsigned int bucket_count = bucket_counter[b];
    if (bucket_count > 1) {
      avx2::quicksort((int*)&new_values[bucket_offsets[b]], bucket_count);
    }
  }
  memcpy(values, new_values, sizeof(int) * count);
  free(new_values);
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
  const int count = 65536 * 32;
  int* keys = (int*)_aligned_malloc(sizeof(int) * count, 32);
  PathSort pathsort;

  for (int i = 0; i < 1; ++i) {
    for (int i = 0; i < count; ++i) {
      keys[i] = random.next() & 0x7FFFFFFF;
     // printf("%u\n", keys[i]);
    }

    unsigned long long now = ticks_now();
    //pathsort.sort_bitonic(keys, nullptr, count);
    //pathsort.radix_sort((unsigned int*)keys, count, 24);
    pathsort.sort_avx_asc(keys, nullptr, count);
    //std::sort(keys, keys + count);
    //simd_merge_sort((float*)keys, count);
    //avx2::quicksort(keys, count);
    printf("%llu ticks\n", ticks_now() - now);

    
    for (unsigned int i = 0; i < count - 1; ++i) {
      //printf("%i\n", keys[i]);
      if (keys[i] > keys[i + 1]) {
   //     printf("FUCK\n");
      }
    }
  }

  _aligned_free(keys);
  return 0;
}