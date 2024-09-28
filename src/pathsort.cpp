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
#define SORTED                0x80
#define SWAP                  0x08

//---
#define ASCENDING(a, b, c, d, e, f, g, h)                                    \
  (((h < 7) << 7) | ((g < 6) << 6) | ((f < 5) << 5) | ((e < 4) << 4) | \
      ((d < 3) << 3) | ((c < 2) << 2) | ((b < 1) << 1) | (a < 0))

//---
#define DESCENDING(a, b, c, d, e, f, g, h)                                    \
  (((a < 0) << 7) | ((b < 1) << 6) | ((c < 2) << 5) | ((d < 3) << 4) | \
      ((e < 4) << 3) | ((f < 5) << 2) | ((g < 6) << 1) | (h < 7))

//---
#define COEX_PERMUTE(vec, a, b, c, d, e, f, g, h, MASK){               \
    __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);  \
    __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask); \
    __m256i min = _mm256_min_epi32(permuted, vec);                     \
    __m256i max = _mm256_max_epi32(permuted, vec);                     \
    constexpr int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
    vec = _mm256_blend_epi32(min, max, blend_mask);}

//---
#define COEX_SHUFFLE(vec, a, b, c, d, e, f, g, h, MASK){               \
    constexpr int shuffle_mask = _MM_SHUFFLE(d, c, b, a);              \
    __m256i shuffled = _mm256_shuffle_epi32(vec, shuffle_mask);        \
    __m256i min = _mm256_min_epi32(shuffled, vec);                     \
    __m256i max = _mm256_max_epi32(shuffled, vec);                     \
    constexpr int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
    vec = _mm256_blend_epi32(min, max, blend_mask);}

//---
#define COEX_SHUFFLE_WITH_INDICES(vec, idx, a, b, c, d, e, f, g, h, MASK){  \
    __m256i oldvec = vec; \
    constexpr int shuffle_mask = _MM_SHUFFLE(d, c, b, a);              \
    __m256i shuffled = _mm256_shuffle_epi32(vec, shuffle_mask);        \
    __m256i shuffled_idx = _mm256_shuffle_epi32(idx, shuffle_mask);    \
    __m256i min = _mm256_min_epi32(shuffled, vec);                     \
    __m256i max = _mm256_max_epi32(shuffled, vec);                     \
    constexpr int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
    vec = _mm256_blend_epi32(min, max, blend_mask); \
    __m256i cmp = _mm256_cmpeq_epi32(vec, oldvec); \
    idx = _mm256_blendv_epi8(shuffled_idx, idx, cmp);}

//---
#define COEX_PERMUTE_WITH_INDICES(vec, idx, a, b, c, d, e, f, g, h, MASK){ \
    __m256i oldvec = vec; \
    __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);  \
    __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask); \
    __m256i permuted_idx = _mm256_permutevar8x32_epi32(idx, permute_mask);    \
    __m256i min = _mm256_min_epi32(permuted, vec);                     \
    __m256i max = _mm256_max_epi32(permuted, vec);                     \
    constexpr int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
    vec = _mm256_blend_epi32(min, max, blend_mask); \
    __m256i cmp = _mm256_cmpeq_epi32(vec, oldvec); \
    idx = _mm256_blendv_epi8(permuted_idx, idx, cmp);}

//---
#define SORT_8_IDX(vec, ORDER, IDX){                                                   \
  COEX_SHUFFLE_WITH_INDICES(vec, IDX, 1, 0, 3, 2, 5, 4, 7, 6, ORDER);                           \
  COEX_SHUFFLE_WITH_INDICES(vec, IDX, 2, 3, 0, 1, 6, 7, 4, 5, ORDER);                           \
  COEX_SHUFFLE_WITH_INDICES(vec, IDX, 0, 2, 1, 3, 4, 6, 5, 7, ORDER);                           \
  COEX_PERMUTE_WITH_INDICES(vec, IDX, 7, 6, 5, 4, 3, 2, 1, 0, ORDER);                           \
  COEX_PERMUTE_WITH_INDICES(vec, IDX, 2, 3, 0, 1, 6, 7, 4, 5, ORDER);                           \
  COEX_PERMUTE_WITH_INDICES(vec, IDX, 1, 0, 3, 2, 5, 4, 7, 6, ORDER);}

//---
#define SORT_8(vec, ORDER){                                                   \
  COEX_SHUFFLE(vec, 1, 0, 3, 2, 5, 4, 7, 6, ORDER);                           \
  COEX_SHUFFLE(vec, 2, 3, 0, 1, 6, 7, 4, 5, ORDER);                           \
  COEX_SHUFFLE(vec, 0, 2, 1, 3, 4, 6, 5, 7, ORDER);                           \
  COEX_PERMUTE(vec, 7, 6, 5, 4, 3, 2, 1, 0, ORDER);                           \
  COEX_SHUFFLE(vec, 2, 3, 0, 1, 6, 7, 4, 5, ORDER);                           \
  COEX_SHUFFLE(vec, 1, 0, 3, 2, 5, 4, 7, 6, ORDER);}

//---
#define SORT_ASCENDING_ALREADY_BITONIC(A){\
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
#define SORT_DESCENDING_ALREADY_BITONIC(A){\
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
PathSort::PathSort()
{
  // Generate original LUT of comparison mask -> permutation indices (1KB).
  for (int m = 0; m < 256; ++m) {
    // Flip the last bit as it should be the opposite of the comparison.
    unsigned int comparison_mask = m ^ (0x1 << 7);
    // Start with an in-order permutation.
    _permute_table[m] = 0x76543210;
    if (comparison_mask == SORTED) {
      continue;
    }
    // While the last element is less than the first, rotate the elements.
    while (comparison_mask & 0x80) {
      _permute_table[m] = (_permute_table[m] << 4) | (_permute_table[m] >> 28);
      comparison_mask = ((comparison_mask << 1) & 0xFF) ^ (0x1 << 7);
    }
    // Wherever an element is greater than its neighbor, move its original index up 
    // and shift other indices down along the way.
    while (comparison_mask) {
      for (int b = 0; b < 8; ++b) {
        if ((comparison_mask >> b) & 0x1) {
          int replace_offset = 1;
          while ((comparison_mask >> (b + replace_offset)) & 0x1) {
            ++replace_offset;
          }
          unsigned int my_index = _permute_table[m] & (0xF << (b << 2));
          for (int i = 0; i < replace_offset; ++i) {
            unsigned int replace = 0xF << ((b + i) << 2);
            unsigned int next = 0xF << ((b + i + 1) << 2);
            unsigned int next_index = _permute_table[m] & next;
            _permute_table[m] = (_permute_table[m] & (~replace)) | (next_index >> 4);
            replace = 0x1 << (b + i);
            next = 0x1 << (b + i + 1);
            unsigned int next_comparison = comparison_mask & next;
            comparison_mask = (comparison_mask & (~replace)) | (next_comparison >> 1);
          }
          unsigned int replace = 0xF << ((b + replace_offset) << 2);
          _permute_table[m] = (_permute_table[m] & (~replace)) | (my_index << (replace_offset << 2));
          break;
        }
      }
    }
  }

  // Place permute table into AVX registers (8KB).
  __m256i select_mask = _mm256_set1_epi32(0x7);
  __m256i shift_right = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  for (int i = 0; i < 256; ++i) {
    unsigned int permutation = _permute_table[i];
    _permute_table_avx[i] = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(permutation),
                                                               shift_right),
                                             select_mask);

    // Pack LUT into as small of bits as possible (768 bytes).
    unsigned int* p = (unsigned int*)&_permute_table_small[i * 3];
    *p = (_permute_table[i] & 0x7) |
         (((_permute_table[i] >> 4) & 0x7) << 3) |
         (((_permute_table[i] >> 8) & 0x7) << 6) |
         (((_permute_table[i] >> 12) & 0x7) << 9) |
         (((_permute_table[i] >> 16) & 0x7) << 12) |
         (((_permute_table[i] >> 20) & 0x7) << 15) |
         (((_permute_table[i] >> 24) & 0x7) << 18) |
           (((_permute_table[i] >> 28) & 0x7) << 21);
  }
}

//---
void PathSort::sort8_ascending(__m256i& a)
{
  /*
  __m256i rotate_left = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
  __m256i rotated_a = _mm256_permutevar8x32_epi32(a,
                                                  rotate_left);
  int comparison = _mm256_movemask_ps(_mm256_castsi256_ps(
                                      _mm256_cmpgt_epi32(a, rotated_a)));
  if (comparison == SORTED) {
    return;
  } else if (comparison == SWAP) {
    a = _mm256_permute2x128_si256(a, a, 0x1);
    return;
  }*/

  SORT_8(a, ASCENDING);

  /*
  int og_comparison = comparison;
  __m256i permutation = _permute_table_avx[comparison];
  a = _mm256_permutevar8x32_epi32(a,
                                  permutation);
  rotated_a = _mm256_permutevar8x32_epi32(a,
                                          rotate_left);
  comparison = _mm256_movemask_ps(_mm256_castsi256_ps(
                                  _mm256_cmpgt_epi32(a, rotated_a)));
  if (comparison == SORTED) {
    return;
  } else if (comparison == SWAP) {
    a = _mm256_permute2x128_si256(a, a, 0x1);
    return;
  } else {
    SORT_8_IDX(a,
               ASCENDING,
               permutation);
    _permute_table_avx[og_comparison] = permutation;
  }*/
}


//---
void PathSort::sort8_descending(__m256i& a)
{
  /*
  __m256i rotate_left = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
  __m256i rotated_a = _mm256_permutevar8x32_epi32(a,
                                                  rotate_left);
  int comparison = _mm256_movemask_ps(_mm256_castsi256_ps(
                                      _mm256_cmpgt_epi32(a, rotated_a)));
  if (comparison == (~SORTED)) {
    return;
  } else if (comparison == (~SWAP)) {
    a = _mm256_permute2x128_si256(a, a, 0x1);
    return;
  }*/

  SORT_8(a, DESCENDING);

  /*
  int og_comparison = comparison;
  __m256i permutation = _permute_table_avx[comparison];
  a = _mm256_permutevar8x32_epi32(a,
                                  permutation);
  rotated_a = _mm256_permutevar8x32_epi32(a,
                                          rotate_left);
  comparison = _mm256_movemask_ps(_mm256_castsi256_ps(
                                  _mm256_cmpgt_epi32(a, rotated_a)));
  if (comparison == SORTED) {
    return;
  } else if (comparison == SWAP) {
    a = _mm256_permute2x128_si256(a, a, 0x1);
    return;
  } else {
    SORT_8_IDX(a,
               ASCENDING,
               permutation);
    _permute_table_avx[og_comparison] = permutation;
  }*/
}

//---
void PathSort::merge16_ascending(__m256i& a,
                                 __m256i& b)
{
  __m256i min = _mm256_min_epi32(a, b);
  int ordered = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(a, min)));
  if (ordered == 0xFF) {
    return;
  } else if (ordered == 0x00) {
    __m256i t = a;
    a = b;
    b = t;
    return;
  }
  b = _mm256_max_epi32(a, b);
  a = min;
  sort8_ascending(a);
  sort8_ascending(b);
}


//---
void PathSort::merge16_descending(__m256i& a,
                                  __m256i& b)
{
  __m256i max = _mm256_max_epi32(a, b);
  int ordered = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(a, max)));
  if (ordered == 0xFF) {
    return;
  } else if (ordered == 0x00) {
    __m256i t = a;
    a = b;
    b = t;
    return;
  }
  b = _mm256_min_epi32(a, b);
  a = max;
  sort8_descending(a);
  sort8_descending(b);
}

//---
void PathSort::sort_bitonic(int* keys,
                    void* values,
                    unsigned int count)
{
  __m256i* _keys = (__m256i*)keys;
  unsigned int level_counts[32] = { 0 };
  unsigned int total_registers = count >> 3;
  for (unsigned int r = 0; r < total_registers; ++r) {
    __m256i reg = _mm256_load_si256(&_keys[r]);
    bool ascending = (r & 0x1) == 0;
    if (ascending) {
      sort8_ascending(reg);
    } else {
      sort8_descending(reg);
    }
    _mm256_store_si256(&_keys[r], reg);
    unsigned int level = 0;
    while (!(++level_counts[level] & 0x1)) {
      ++level;
      const unsigned int batch_registers = 0x1 << (level - 1);
      ascending = (r & (0x1 << level)) == 0;
      __m256i* left = &_keys[(r >> level) << level];
      __m256i* right = left + batch_registers;
      if (level == 1) {
        if (ascending) {
          merge16_ascending(*left, *right);
        } else {
          merge16_descending(*left, *right);
        }
      } else {
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
            sort8_ascending(L);
            sort8_ascending(R);
          } else {
            sort8_descending(L);
            sort8_descending(R);
          }
          _mm256_store_si256(left + f, L);
          _mm256_store_si256(left + f + 1, R);
        }
      }
    }
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