#include <stdio.h>
#include "src/pathsort.h"

//---
#define SORTED 0x80

//---
#define ASC(a, b, c, d, e, f, g, h)                                    \
  (((h < 7) << 7) | ((g < 6) << 6) | ((f < 5) << 5) | ((e < 4) << 4) | \
      ((d < 3) << 3) | ((c < 2) << 2) | ((b < 1) << 1) | (a < 0))

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
#define SORT_8(vec, IDX){                                                   \
  COEX_SHUFFLE_WITH_INDICES(vec, IDX, 1, 0, 3, 2, 5, 4, 7, 6, ASC);                           \
  COEX_SHUFFLE_WITH_INDICES(vec, IDX, 2, 3, 0, 1, 6, 7, 4, 5, ASC);                           \
  COEX_SHUFFLE_WITH_INDICES(vec, IDX, 0, 2, 1, 3, 4, 6, 5, 7, ASC);                           \
  COEX_PERMUTE_WITH_INDICES(vec, IDX, 7, 6, 5, 4, 3, 2, 1, 0, ASC);                           \
  COEX_PERMUTE_WITH_INDICES(vec, IDX, 2, 3, 0, 1, 6, 7, 4, 5, ASC);                           \
  COEX_PERMUTE_WITH_INDICES(vec, IDX, 1, 0, 3, 2, 5, 4, 7, 6, ASC);}

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
void PathSort::sort(int* array,
                    unsigned int count)
{
  __m256i rotate_left = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
  // Sort 8-wide in each batch.
  unsigned int batches = count / 8;
  for (unsigned int b = 0; b < batches; ++b) {
    __m256i batch = _mm256_load_si256((__m256i*)&array[b << 3]);
    __m256i rotated_batch = _mm256_permutevar8x32_epi32(batch,
                                                        rotate_left);
    int comparison = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(batch,
                                                                               rotated_batch)));
    int og_comparison = comparison;
    __m256i permutation = _permute_table_avx[comparison];
    batch = _mm256_permutevar8x32_epi32(batch,
                                        permutation);
    rotated_batch = _mm256_permutevar8x32_epi32(batch,
                                                rotate_left);
    comparison = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(batch,
                                                                           rotated_batch)));
    if (comparison != SORTED) {
      SORT_8(batch,
             permutation);
      _permute_table_avx[og_comparison] = permutation;
    }
    _mm256_store_si256((__m256i*)&array[b << 3],
                       batch);
  }
}

int main()
{
  int* values = (int*)_aligned_malloc(sizeof(int) * 16, 32);
  for (int i = 0; i < 16; ++i) {
    values[i] = 16 - i;
  }

  PathSort pathsort;
  pathsort.sort(values, 16);

  for (int i = 0; i < 16; ++i) {
    printf("%u\n", values[i]);
  }
  _aligned_free(values);

  return 0;
}