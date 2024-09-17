#include <stdio.h>
#include <immintrin.h>
#include "src/pathsort.h"

int main()
{
  __m256i values = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  SORT_8(values);

  alignas(32) int v[8];
  _mm256_store_si256((__m256i*)v, values);
  for (int i = 0; i < 8; ++i) {
    printf("%u", v[i]);
  }

  return 0;
}