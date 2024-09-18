#pragma once
#include <immintrin.h>
#include <avxintrin.h>
#include <avx2intrin.h>

//---
struct PathSort
{
  //---
  unsigned int _permute_table[256];
  char         _permute_table_small[768];
  __m256i      _permute_table_avx[256];

  //---
  PathSort();
  ~PathSort() {}
  void sort(int* array,
            unsigned int count);
};