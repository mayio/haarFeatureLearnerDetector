/*
 *
 *  Created on: May 17, 2017
 *      Author: Mario Lüder
 *
 */


#include "FeatureTypes.cuh"

#include <sstream>
#include <assert.h>
#include "utilities.cuh"

#define FEATURE_DATA_MAX_SIZE 31 * 1024
__constant__ uint8_t g_FeatureData[FEATURE_DATA_MAX_SIZE];

FeatureTypes::~FeatureTypes()
{
   if (data != NULL)
   {
      delete [] data;
      data = NULL;
   }

   if (gpuData != NULL)
   {
      CUDA_CHECK_RETURN(cudaFree(gpuData));
      gpuData = NULL;
   }
}

void FeatureTypes::generateClassifier(const double scale, const uint32_t windowWidth,
         const uint32_t windowHeight, bool calcOnlySize, uint32_t & memsize)
{
   memsize = 0;
   assert(data || calcOnlySize);

   std::vector<uint32_t> featureTypeOffsets;

   // count of feature types
   memsize += sizeof(uint32_t);

   // calculate the size first
   uint32_t countFeatureTypes = this->size();

   if (!calcOnlySize)
      *(uint32_t*) (data) = countFeatureTypes;

   const uint32_t sizeFeatureTypeOffsets = countFeatureTypes
         * sizeof(uint32_t);
   memsize += sizeFeatureTypeOffsets;

   for (uint32_t featureTypeIdx = 0; featureTypeIdx < countFeatureTypes;
         ++featureTypeIdx)
   {
#ifdef DEBUG
      // std::cout << "Debug: featureTypeIdx:" << featureTypeIdx << std::endl;
#endif
      // store the offset of each feature type
      featureTypeOffsets.push_back(memsize);

      // header size for [ feature width, feature height, feature count ]
      const uint32_t headerSize = 3 * sizeof(uint32_t);

      const FeatureType & featureType = at(featureTypeIdx);
      const uint32_t featureHeightPx = featureType.mFeatureHeight
            * featureType.mRect.height;
      const uint32_t featureWidthPx = featureType.mFeatureWidth
            * featureType.mRect.width;
      const int32_t windowHeightMax = windowHeight - featureHeightPx;
      const int32_t windowWidthMax = windowWidth - featureWidthPx;

      // assure the window size is big enough
      assert(windowHeightMax > 0 && windowWidthMax > 0);

      // calculate how many feature can be generated in x direction
      //
      // the feature is scaled by
      //   scaledWidth = featureWidthPx * scale^n
      // this is under the condition
      //   scaledWidth <= windowWidthMax
      //   n := times scale
      //
      // n is determined by
      //   n = log(windowWidthMax/featureWidthPx) / log(scale)
      //
      // the same is done with height
      //
      const uint32_t nWidthScales = (scale > 1.0) ? ((uint32_t) (log(
            windowWidthMax / featureWidthPx) / log(scale))) : 1.0;
      const uint32_t nHeightScales = (scale > 1.0) ? ((uint32_t) (log(
            windowHeightMax / featureHeightPx) / log(scale))) : 1.0;

      const uint32_t countRectangles = featureType.mFeatureHeight
            * featureType.mFeatureWidth;
      const uint32_t countClassifier = nWidthScales * nHeightScales;

      // make sure that this value is the same as the number of feature types
      assert(countRectangles == featureType.mTypes.size());

      // the size of width, height and type - see FeatureRectangle
      const uint32_t rectangleValuesSize = 3 * sizeof(int32_t);

      uint32_t offset = memsize;

      if (!calcOnlySize)
      {
         // write header
         *(uint32_t*) (data + offset) = featureType.mFeatureWidth;
         offset += sizeof(uint32_t);
         *(uint32_t*) (data + offset) = featureType.mFeatureHeight;
         offset += sizeof(uint32_t);

         // this data will change as we do not store all classifiers (because of rounding)
         //*(uint32_t*)(data + offset) = countClassifier;            offset += sizeof(uint32_t);
         // instead, we remember the offset of the count variable
         uint32_t offsetCountClassifier = offset;
         offset += sizeof(uint32_t);
         uint32_t countStoredClassifier = 0;

         Scale previousRowScale(0, 0);

         for (uint32_t heightScaleIdx = 0; heightScaleIdx < nHeightScales;
               ++heightScaleIdx)
         {
            Scale previousColumnScale = previousRowScale;

            // scale the rectangle
            const uint32_t scaledRectangleHeight =
                  (uint32_t) (featureType.mRect.height
                        * pow(scale, heightScaleIdx));

            if (scaledRectangleHeight == previousRowScale.y)
            {
               continue;
            }

            previousRowScale = Scale(0, scaledRectangleHeight);

            for (uint32_t widthScaleIdx = 0; widthScaleIdx < nWidthScales;
                  ++widthScaleIdx)
            {
               const uint32_t scaledRectangleWidth =
                     (uint32_t) (featureType.mRect.width
                           * pow(scale, widthScaleIdx));

               if (previousColumnScale
                     != Scale(widthScaleIdx, scaledRectangleHeight))
               {
                  // store the scales for each each rectangle
                  for (uint32_t rectangleIdx = 0;
                        rectangleIdx < countRectangles; ++rectangleIdx)
                  {
                     *(uint32_t*) (data + offset) = scaledRectangleWidth;
                     offset += sizeof(uint32_t);
                     *(uint32_t*) (data + offset) = scaledRectangleHeight;
                     offset += sizeof(uint32_t);
                     *(int32_t*) (data + offset) =
                           featureType.mTypes[rectangleIdx];
                     offset += sizeof(int32_t);
                  }

                  previousColumnScale = Scale(widthScaleIdx,
                        scaledRectangleHeight);
                  countStoredClassifier++;
               }
            }
         }
#ifdef DEBUG
//         std::cout << "Debug: Count Stored Classifier:"
//               << countStoredClassifier << std::endl;
#endif

         // store the classifier count
         *(uint32_t*) (data + offsetCountClassifier) = countStoredClassifier;
         memsize += headerSize
               + countRectangles * countStoredClassifier
                     * rectangleValuesSize;
      }
      else
      {
         memsize += headerSize
               + countRectangles * countClassifier * rectangleValuesSize;
      }

      if (!calcOnlySize)
      {
         assert(offset == memsize);
      }
   }

   if (!calcOnlySize)
   {
      // store the offsets of the feature types
      for (uint32_t featureTypeOffsetIdx = 0;
            featureTypeOffsetIdx < featureTypeOffsets.size();
            ++featureTypeOffsetIdx)
      {
         *(uint32_t*) (data + sizeof(uint32_t)
               + featureTypeOffsetIdx * sizeof(uint32_t)) =
               featureTypeOffsets[featureTypeOffsetIdx];
      }
   }
}

void FeatureTypes::generateClassifier(const double scale, const uint32_t windowWidth,
      const uint32_t windowHeight, bool copyToConst)
{
   if (data != NULL)
   {
      delete [] data;
      data = NULL;
   }

   if (gpuData != NULL)
   {
      CUDA_CHECK_RETURN(cudaFree(gpuData));
      gpuData = NULL;
   }

   dataSize = 0;

   // calc first size
   uint32_t maxSize = 0;
   generateClassifier(scale, windowWidth, windowHeight, true, maxSize);
#ifdef DEBUG
   std::cout << "Debug: generateClassifier estimated size:" << maxSize
         << std::endl;
#endif
   data = new uint8_t[maxSize];
   assert(data);
   uint32_t usedSize = 0;
   generateClassifier(scale, windowWidth, windowHeight, false, usedSize);
#ifdef DEBUG
   std::cout << "Debug: generateClassifier used size:" << usedSize
         << std::endl;
#endif
   CUDA_CHECK_RETURN(
         cudaMalloc((void ** )&gpuData, usedSize));

   CUDA_CHECK_RETURN(
         cudaMemcpy(gpuData, data, usedSize, cudaMemcpyHostToDevice));

   dataSize = usedSize;

   if (copyToConst)
   {
      copyToConstantMemory();
   }
}

void FeatureTypes::copyToConstantMemory()
{
   if (gpuData)
   {
      assert(dataSize <= FEATURE_DATA_MAX_SIZE);
      CUDA_CHECK_RETURN(cudaMemcpyToSymbol(g_FeatureData, gpuData, dataSize));
      CUDA_CHECK_RETURN(cudaFree(gpuData));
      gpuData = NULL;
   }
}

uint8_t * FeatureTypes::getConstantFeatureData()
{
   uint8_t * constFeatureData = NULL;
   CUDA_CHECK_RETURN(cudaGetSymbolAddress((void **)(&constFeatureData), g_FeatureData));
   return constFeatureData;
}
