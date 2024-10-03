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
static void merge_split(int* left,
                        int* right,
                        unsigned int compare_count,
                        unsigned int total_count,
                        bool ascending)
{

  // Find the bitonic point with binary search.
  unsigned int step_size = compare_count >> 1;
  unsigned int bitonic_point = step_size;
  step_size >>= 1;
  int L = left[bitonic_point];
  int R = right[bitonic_point];
  int comparison = L > R;
  while (step_size > 0) {
    bitonic_point += !comparison * step_size;
    bitonic_point -= comparison * step_size;
    int L = left[bitonic_point];
    int R = right[bitonic_point];
    comparison = L > R;
    step_size >>= 1;
  }
  bitonic_point += !comparison;

  // When ascending, we will always get L ascending and R descending.
  // When descending, we will always get L descending and R ascending.

  /*
  \    /
   \  /
    \/
    
    /\
   /  \
  /    \*/

  // If swap_ascending, we swap L and R, so swap ascending always becomes true.
  // If ascending we swap after, descending we swap before.

  // If ascending but swap_descending, we swap before the point.
  // If ascending but swap_ascending, we swap after the point
  // If descending but swap_descending, we swap after the point
  // If descending but swap_asending, we swap before the point.


  unsigned int swap_left = ascending ? bitonic_point : 0;
  unsigned int swap_right = ascending ? total_count : (bitonic_point + (bitonic_point < total_count));

  // Perform the necessary swaps around the bitonic point.
  for (unsigned int n = swap_left; n < swap_right; ++n) {
    int t = left[n];
    left[n] = right[n];
    right[n] = t;
  }

  // We need to swap left and right 
  unsigned int left_count = bitonic_point;
  unsigned int right_count = total_count - bitonic_point;
  merge_split(ascending ? left : right, ascending ? right : left, left_count, left_count + right_count, ascending);
  merge_split(ascending ? right : left, ascending ? left : right, left_count, left_count + right_count, ascending);
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
    printf("%i, ", keys[i]);
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
      
      // We want to recursively merge an ascending/descending, to create 2 new bitonic sequences,
      // which we will then split at their bitonic points to create a new ascending/descending pair 
      // to merge and split.
      // This should continue recursively until the bitonic point creates a bitonic sequence that is
      // 8 or less elements, which we sort, while the other half should already be sorted as we want.
      merge_split(left, right, batch_elements, batch_elements << 1, true);
    } while (!(++level_counts[level] & 0x1));
  }

  /*
  for (unsigned int r = 0; r < total_registers; r += 4) {
    unsigned int level = 1;
    do {
      ++level;
      const unsigned int batch_registers = 0x1 << (level - 1);
      int ascending = (r & (0x1 << level)) == 0;
      __m256i* left = &_keys[(r >> level) << level];
      __m256i* right = left + batch_registers;
      const unsigned int registers = 0x1 << level;
      for (unsigned int descent = level; descent > 0; --descent) {
        const unsigned int splits = registers >> descent;
        const unsigned int split_registers = 0x1 << (descent - 1);
        const unsigned int split_elements = split_registers << 3;
        for (unsigned int s = 0; s < splits; ++s) {
          int* nleft = (int*)&left[s << descent];
          int* nright = (int*)(nleft + split_elements);
          for (unsigned int n = 0; n < split_elements; ++n) {
            int a = nleft[n];
            int b = nright[n];
            int min = a <= b ? a : b;
            int max = a > b ? a : b;
            nleft[n] = ascending ? min : max;
            nright[n] = ascending ? max : min;
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
  }*/

  /*
  for (unsigned int r = 0; r < total_registers; r += 4) {
    unsigned int level = 1;
    do {
      ++level;
      const unsigned int batch_registers = 0x1 << (level - 1);
      int ascending = (r & (0x1 << level)) == 0;
      __m256i* left = &_keys[(r >> level) << level];
      __m256i* right = left + batch_registers;
      const unsigned int registers = 0x1 << level;
      for (unsigned int descent = level; descent > 0; --descent) {
        const unsigned int splits = registers >> descent;
        const unsigned int split_registers = 0x1 << (descent - 1);
        const unsigned int split_elements = split_registers << 3;
        for (unsigned int s = 0; s < splits; ++s) {
          int swap_ascending = (descent == level) ? 1 : ((s + ascending) & 0x1);
          int* nleft = (int*)&left[s << descent];
          int* nright = (int*)(nleft + split_elements);
          // Find the bitonic point with binary search.
          unsigned int step_size = split_registers << 2;
          unsigned int bitonic_point = step_size;
          step_size >>= 1;
          int L = nleft[bitonic_point];
          int R = nright[bitonic_point];
          int comparison = swap_ascending ? (L > R) : (R > L);
          while (step_size > 0) {
            bitonic_point += !comparison * step_size;
            bitonic_point -= comparison * step_size;
            int L = nleft[bitonic_point];
            int R = nright[bitonic_point];
            comparison = swap_ascending ? (L > R) : (R > L);
            step_size >>= 1;
          }
          bitonic_point += !comparison;
          int swap_left = ascending ^ swap_ascending;
          int left_side = swap_left ? 0 : bitonic_point;
          int right_side = swap_left ? (bitonic_point + (bitonic_point < split_elements)) : split_elements;

          for (unsigned int n = left_side; n < right_side; ++n) {
            int a = nleft[n];
            int b = nright[n];
            int min = a <= b ? a : b;
            int max = a > b ? a : b;
            nleft[n] = ascending ? min : max;
            nright[n] = ascending ? max : min;
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
  }*/

  /*
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
  }*/

  /*
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
          bool split_ascending = (s & 0x1) == 0;
          bool swap_ascending = split_ascending ^ ascending;
          unsigned int step_size = split_registers >> 1;
          unsigned int bitonic_point = 0; // step_size;
          __m256i* nleft = &left[s << descent];
          __m256i* nright = nleft + split_registers;
          __m256i L = nleft[bitonic_point];
          __m256i R = nright[bitonic_point];
          if (split_registers > 1) {

            // Find the bitonic point using binary search.
            __m256i* bleft = nleft;
            __m256i* bright = nright;
            int compare = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(R, L)));
            while (((compare == 0x00) || (compare == 0xFF)) &&
                   (step_size > 0)) {
              bool move_left = (compare == 0x00);
              bool move_right = (compare == 0xFF);
              bitonic_point -= move_left * step_size;
              bitonic_point += move_right * step_size;
              step_size >>= 1;
              L = nleft[bitonic_point];
              R = nright[bitonic_point];
              compare = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(R, L)));
            }
            // If search ended without a mixed register, the bitonic point is the first element of the next register.
            if (compare == 0x00) {
              bitonic_point += (bitonic_point + 1) < split_registers;
              L = nleft[bitonic_point];
              R = nright[bitonic_point];
            }

            // Find the bitonic point using linear search.
            int compare_against = split_ascending ? 0x00 : 0xFF;
            for (unsigned int i = 0; i < split_registers; ++i) {
              L = nleft[i];
              R = nright[i];
              int compare = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(L, R)));
              if (compare != compare_against) {
                bitonic_point = i;
                break;
              }
            }
          }
          // Now that we've found the bitonic point, do the necessary swaps.
          __m256i min = _mm256_min_epi32(L, R);
          __m256i max = _mm256_max_epi32(L, R);
          _mm256_store_si256(&nleft[bitonic_point], ascending ? min : max);
          _mm256_store_si256(&nright[bitonic_point], ascending ? max : min);
          int left = swap_ascending ? 0 : (bitonic_point + 1);
          int right = swap_ascending ? bitonic_point : split_registers;
          int count = right - left;
          for (int s = 0; s < count; ++s) {
            __m256i L = nleft[left + s];
            __m256i R = nright[left + s];
            _mm256_store_si256(&nleft[left + s], R);
            _mm256_store_si256(&nright[left + s], L);
          }
        }
      }
      for (unsigned int f = 0; f < registers; ++f) {
        __m256i L = _mm256_load_si256(left + f);
        if (ascending) {
          SORT8(L, true);
        } else {
          SORT8(L, false);
        }
        _mm256_store_si256(left + f, L);
      }
    } while (!(++level_counts[level] & 0x1));
  }*/
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
  const int count = 64;
  int* keys = (int*)_aligned_malloc(sizeof(int) * count, 32);
  PathSort pathsort;

  int v[64] =
  {
    0, 1, 2, 2, 2, 2, 3, 3, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 5, 5, 5, 5, 3, 3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7, 7, 7, 7, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 0
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
    //  printf("%i\n", keys[i]);
      if (v[i] > v[i + 1]) {
        printf("FUCK\n");
      }
    }
  }

  _aligned_free(keys);
  return 0;
}