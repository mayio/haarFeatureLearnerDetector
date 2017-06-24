/*
 *
 *  Created on: May 17, 2017
 *      Author: Mario Lüder
 *
 */


#ifndef FEATUREVALUES_H_
#define FEATUREVALUES_H_

#include "stdint.h"
#include "assert.h"

struct GetFeatureValueResult
{
   int32_t featureValue;
   int32_t imageType;
   uint32_t imageIdx;
};

class FeatureValues
{
public:
   __device__ FeatureValues(uint32_t arraySize) :
         data(NULL), sortedIndex(NULL), count(arraySize)
   {
      init();
   }

   __device__ ~FeatureValues()
   {
      clear();
   }

   __forceinline__ __device__ void init()
   {
      data = new int32_t[count * 4];
      assert(data);
      sortedIndex = (uint32_t*) (&data[count]);
      sortedDataCpy = (uint32_t*) (&data[count * 2]);
      sortedIndexCpy = (uint32_t*) (&data[count * 3]);

      for (uint32_t i = 0; i < count; ++i)
      {
         data[i] = INT_MAX;
         sortedIndex[i] = i;
      }

      isInit = true;
   }

   __forceinline__ __device__ void reset()
   {
      for (uint32_t i = 0; i < count; ++i)
      {
         data[i] = INT_MAX;
         sortedIndex[i] = i;
      }
   }

   __forceinline__ __device__ void clear()
   {
      if (isInit)
      {
         delete[] data;
         data = NULL;
         sortedIndex = NULL;
         sortedDataCpy = NULL;
         sortedIndexCpy = NULL;
         isInit = false;
      }
   }

   // access to the sorted array
   __forceinline__ __device__     int32_t operator[](uint32_t i)
   {
      assert(isInit);
      return data[sortedIndex[i]];
   }

   __forceinline__  __device__  int32_t * getData() const
   {
      assert(isInit);
      return data;
   }
   __forceinline__ __device__     uint32_t getCount() const
   {
      assert(isInit);
      return count;
   }
   __forceinline__ __device__     uint32_t * getSortedIdx() const
   {
      assert(isInit);
      return sortedIndex;
   }

   __device__ void sort()
   {
      assert(isInit);
      // from
      // http://stackoverflow.com/questions/1271367/radix-sort-implemented-in-c
      size_t index[4][256];            // count / index matrix

      uint32_t * dataCpy = (uint32_t*) data; // new uint32_t[count];

      // create an index
      // convert signed to unsigned
      for (uint32_t i = 0; i < count; ++i)
      {
         sortedIndex[i] = i;
         dataCpy[i] = (int64_t) (data[i]) + INT_MIN;
      }

      for (uint32_t i = 0; i < 4; ++i)
      {
         for (uint32_t j = 0; j < 256; ++j)
         {
            index[i][j] = 0;
         }
      }

      uint32_t * b = sortedDataCpy;

      size_t i, j, m, n, idx;
      uint32_t u;

      // generate histogram
      for (i = 0; i < count; i++)
      {
         u = dataCpy[i];

         for (j = 0; j < 4; j++)
         {
            index[j][(size_t) (u & 0xff)]++;
            u >>= 8;
         }
      }

      // convert to indices
      for (j = 0; j < 4; j++)
      {
         m = 0;
         for (i = 0; i < 256; i++)
         {
            n = index[j][i];
            index[j][i] = m;
            m += n;
         }
      }
      for (j = 0; j < 4; j++)
      {             // radix sort
         for (i = 0; i < count; i++)
         {     //  sort by current lsb
            u = dataCpy[i];
            m = (size_t) (u >> (j << 3)) & 0xff;

            // store the index and not the value
            idx = index[j][m]++;
            b[idx] = u;
            sortedIndexCpy[idx] = sortedIndex[i];
         }

         // swap data
         uint32_t * tmpPtr = b;
         b = dataCpy;
         dataCpy = tmpPtr;

         // swap index
         tmpPtr = sortedIndexCpy;
         sortedIndexCpy = sortedIndex;
         sortedIndex = tmpPtr;
      }

      // convert unsigned to signed
      for (uint32_t i = 0; i < count; ++i)
      {
         data[i] = (int64_t) (dataCpy[i]) - INT_MIN;
      }
   }

private:
   int32_t * data;
   uint32_t * sortedIndex;
   uint32_t * sortedDataCpy;
   uint32_t * sortedIndexCpy;
   const uint32_t count;
   bool isInit;
};

#endif /* FEATUREVALUES_H_ */
