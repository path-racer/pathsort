#include <stdio.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <algorithm>
#include "src/pathsort.h"
#include "avx2sort.h"

//---
#define SORTED                0x80
#define SWAPPED               0x01
#define MERGE_SORTED          0x00
#define MERGE_SWAPPED         0x01
#define MERGE_NORMAL          0x02

//---
#define ASC(a, b, c, d, e, f, g, h)                                    \
  (((h < 7) << 7) | ((g < 6) << 6) | ((f < 5) << 5) | ((e < 4) << 4) | \
      ((d < 3) << 3) | ((c < 2) << 2) | ((b < 1) << 1) | (a < 0))

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
#define SORT_8_IDX(vec, IDX){                                                   \
  COEX_SHUFFLE_WITH_INDICES(vec, IDX, 1, 0, 3, 2, 5, 4, 7, 6, ASC);                           \
  COEX_SHUFFLE_WITH_INDICES(vec, IDX, 2, 3, 0, 1, 6, 7, 4, 5, ASC);                           \
  COEX_SHUFFLE_WITH_INDICES(vec, IDX, 0, 2, 1, 3, 4, 6, 5, 7, ASC);                           \
  COEX_PERMUTE_WITH_INDICES(vec, IDX, 7, 6, 5, 4, 3, 2, 1, 0, ASC);                           \
  COEX_PERMUTE_WITH_INDICES(vec, IDX, 2, 3, 0, 1, 6, 7, 4, 5, ASC);                           \
  COEX_PERMUTE_WITH_INDICES(vec, IDX, 1, 0, 3, 2, 5, 4, 7, 6, ASC);}

//---
#define SORT_8(vec){                                                   \
  COEX_SHUFFLE(vec, 1, 0, 3, 2, 5, 4, 7, 6, ASC);                           \
  COEX_SHUFFLE(vec, 2, 3, 0, 1, 6, 7, 4, 5, ASC);                           \
  COEX_SHUFFLE(vec, 0, 2, 1, 3, 4, 6, 5, 7, ASC);                           \
  COEX_PERMUTE(vec, 7, 6, 5, 4, 3, 2, 1, 0, ASC);                           \
  COEX_SHUFFLE(vec, 2, 3, 0, 1, 6, 7, 4, 5, ASC);                           \
  COEX_SHUFFLE(vec, 1, 0, 3, 2, 5, 4, 7, 6, ASC);}

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
void PathSort::sort8(__m256i& array)
{
  __m256i rotate_left =   _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
  __m256i rotated_array = _mm256_permutevar8x32_epi32(array,
                                                      rotate_left);
  int comparison =        _mm256_movemask_ps(_mm256_castsi256_ps(
                                             _mm256_cmpgt_epi32(array,
                                                                rotated_array)));
  if (comparison != SORTED) {
    __m256i permutation = _permute_table_avx[comparison];
    array =         _mm256_permutevar8x32_epi32(array,
                                                permutation);
    rotated_array = _mm256_permutevar8x32_epi32(array,
                                                rotate_left);
    comparison =    _mm256_movemask_ps(_mm256_castsi256_ps(
                                       _mm256_cmpgt_epi32(array,
                                                          rotated_array)));
    if (comparison != SORTED) {
      SORT_8(array);
    }
  }
}

//---
void PathSort::sort8_adaptive(__m256i& array)
{
  __m256i rotate_left =   _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
  __m256i rotated_array = _mm256_permutevar8x32_epi32(array,
                                                      rotate_left);
  int comparison =        _mm256_movemask_ps(_mm256_castsi256_ps(
                                             _mm256_cmpgt_epi32(array,
                                                                rotated_array)));
  if (comparison != SORTED) {
    int og_comparison = comparison;
    __m256i permutation = _permute_table_avx[comparison];
    array =         _mm256_permutevar8x32_epi32(array,
                                                permutation);
    rotated_array = _mm256_permutevar8x32_epi32(array,
                                                rotate_left);
    comparison =    _mm256_movemask_ps(_mm256_castsi256_ps(
                                       _mm256_cmpgt_epi32(array,
                                                          rotated_array)));
    if (comparison != SORTED) {
      SORT_8_IDX(array,
                 permutation);
      _permute_table_avx[og_comparison] = permutation;
    }
  }
}


//---
int PathSort::merge16(__m256i& a,
                      __m256i& b)
{
  __m256i reverse =    _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  __m256i reversed_b = _mm256_permutevar8x32_epi32(b,
                                                   reverse);
  __m256i gt = _mm256_cmpgt_epi32(reversed_b, a);
  __m256i eq = _mm256_cmpeq_epi32(reversed_b, a);
  int gteq = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_or_si256(gt, eq)));
  int lteq = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_xor_si256(gt, _mm256_set1_epi32(-1))));
  if (gteq & SORTED) {
    return MERGE_SORTED;
  } else if (lteq & SWAPPED) {
    __m256i t = a;
    a = b;
    b = t;
    return MERGE_SWAPPED;
  }
  b = _mm256_blendv_epi8(a, reversed_b, gt);
  a = _mm256_blendv_epi8(reversed_b, a, gt);
  sort8(a);
  sort8(b);
  return MERGE_NORMAL;
}

//---
void PathSort::sort(int* array,
                    unsigned int count)
{
  // Sort 8-wide in each batch.
  unsigned int batches = count >> 3;
  for (unsigned int b = 0; b < batches; ++b) {
    __m256i batch = _mm256_load_si256((__m256i*) & array[b << 3]);
    sort8(batch);
    _mm256_store_si256((__m256i*)&array[b << 3],
                       batch);
  }
  // Merge pairs downward.
  int level = 0;
  unsigned int merges = count >> (4 + level);
  while (merges > 0) {
    const unsigned int merge_size = 0x8 << level;
    const unsigned int batch_count = merge_size >> 3;
    for (unsigned int m = 0; m < merges; ++m) {
      __m256i* left = (__m256i*)&array[(m << (4 + level))];
      __m256i* right = (__m256i*)&array[((m << (4 + level)) + merge_size)];
      __m256i* end = right + batch_count;
      __m256i R = _mm256_load_si256(right);
      while (left < right) {
        __m256i L = _mm256_load_si256(left);
        __m256i R = _mm256_load_si256(right);
        if (merge16(L, R) != MERGE_SORTED) {
          _mm256_store_si256(left, L);
          _mm256_store_si256(right, R);
          if ((right + 1) < end) {
            __m256i* n = right;
            __m256i T = _mm256_load_si256(n + 1);
            while (merge16(R, T) != MERGE_SORTED) {
              _mm256_store_si256(n, R);
              _mm256_store_si256(++n, T);
              if ((n + 1) == end) {
                break;
              }
              R = T;
              T = _mm256_load_si256(n + 1);
            }
          }
        }
        ++left;
      }
    }
    merges = count >> (4 + (++level));


    /*
    for (int i = 0; i < count; ++i) {
      printf("%i\n", array[i]);
    }
    printf("---\n");*/
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
  int* values = (int*)_aligned_malloc(sizeof(int) * count, 32);

  PathSort pathsort;
  for (int i = 0; i < 10; ++i) {
    for (int i = 0; i < count; ++i) {
      values[i] = random.next() & 0xFFFFFFFF;
    }
    unsigned long long now = ticks_now();
    pathsort.sort(values, count);
    //std::sort(values, values + count);
    printf("%llu ticks\n", ticks_now() - now);
  }

  for (int i = 0; i < count - 1; ++i) {
    if (values[i] > values[i + 1]) {
      printf("FUCK\n");
      return 0;
    }
  }

  _aligned_free(values);
  return 0;
}