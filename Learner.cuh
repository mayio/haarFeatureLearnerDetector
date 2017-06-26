/*
 *
 *  Created on: May 17, 2017
 *      Author: Mario Lüder
 *
 */

#ifndef LEARNER_H_
#define LEARNER_H_

#define WORK_SIZE 192

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// load image includes
#include <opencv2/core/core.hpp>
#include <opencv2/core/gpumat.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_profiler_api.h>

#include "FeatureValues.cuh"
#include "Classifier.cuh"
#include "FeatureTypes.cuh"
#include "defines.cuh"

texture<int2, 1> texAllImageWeights;

static __forceinline__ __device__ double fetch_double(texture<int2, 1> t, int i)

{
   int2 v = tex1Dfetch(t, i);
   return __hiloint2double(v.y, v.x);
}

__forceinline__ __device__ bool getFeatureValues(
      const int32_t * integralImages,
      const uint32_t availableImages[],
      const uint32_t countImages,
      const uint32_t imageCols,
      const uint32_t imageRows,
      const uint8_t * const classifierData, const uint32_t classifierScaleIdx,
      const uint32_t classifierWidth, const uint32_t classifierHeight,
      const uint32_t x, const uint32_t y, FeatureValues & featureValues)
{
   bool available = false;
   uint32_t rectWidth;
   uint32_t rectHeight;
   const uint8_t * singleClassifier = NULL;

   Classifier::getClassifierScale(classifierData, classifierScaleIdx,
         &singleClassifier, rectWidth, rectHeight);
   assert(singleClassifier);

   // rect width and height must be at least 2!
   assert(rectWidth > 1);
   assert(rectHeight > 1);

   const uint32_t classifierRightPoint = x + rectWidth * classifierWidth;
   const uint32_t classifierBottomPoint = y + rectHeight * classifierHeight;
   const bool outOfRange =
         ((classifierRightPoint <= imageCols)
               && (classifierBottomPoint <= imageRows)) ? false : true;

   int32_t * featureValuesData = featureValues.getData();

   if (!outOfRange)
   {
      for (uint32_t imageIdx = 0; imageIdx < countImages; ++imageIdx)
      {
         const int32_t * integralImage = integralImages + imageCols * imageRows * availableImages[imageIdx];
         Classifier::getFeatureValue(integralImage, singleClassifier, imageCols, x, y,
               rectWidth, rectHeight, classifierWidth, classifierHeight,
               featureValuesData[imageIdx]);
      }
      available = true;
   }
   else
   {
      for (uint32_t imageIdx = 0; imageIdx < countImages; ++imageIdx)
      {
         featureValuesData[imageIdx] = INT_MAX;
      }
   }

   return available;
}

__forceinline__ __device__ bool getThresholdAndPolarity(
      FeatureValues & featureValues,
      // const double * const imageWeights,
      const double Tp, const double Tn, const uint32_t positiveImages,
      int32_t & threshold, int32_t & polarity)
{
   bool available = false;

   // Tp : total sum of of positive example weights
   // Tn : total sum of of negative example weights
   // Sp : sum of positive weights below the current example
   // Sn : sum of negative weights below the current example
   // e = min( Sp + (Tn - Sn), Sn + (Tp - Sp) )
   //
   // return the threshold with the lowest 'e'
   //double Tp = 0.0;
   //double Tn = 0.0;
   double Sp = 0.0;
   double Sn = 0.0;

   const uint32_t countFeatureValues = featureValues.getCount();
   featureValues.sort();
   const uint32_t * const sortedIndex = featureValues.getSortedIdx();
   const int32_t * const featureValuesData = featureValues.getData();

   /*
    for (uint32_t i = 0; i < countFeatureValues; ++i)
    {
    const uint32_t imageIdx = sortedIndex[i];

    if (imageIdx < positiveImages)
    {
    // if positive Image, add image weight
    Tp += imageWeights[imageIdx];
    }
    else
    {
    // if negative Image, add image weight
    Tn += imageWeights[imageIdx];
    }
    }
    */

   double eMin = DBL_MAX;
   threshold = INT_MAX;
   polarity = -1;

   for (uint32_t i = 0; i < countFeatureValues; ++i)
   {
      const int32_t featureValue = featureValuesData[i];

      if (featureValue == INT_MAX)
      {
         continue;
      }

      available = true;
      const uint32_t imageIdx = sortedIndex[i];

      const double e1 = Sp + (Tn - Sn);
      const double e2 = Sn + (Tp - Sp);
      const double e = (e1 < e2) ? e1 : e2;

      if (e < eMin)
      {
         eMin = e;
         threshold = featureValue;
         polarity = (e1 < e2) ? -1 : 1;
      }

      const double imageWeight = fetch_double(texAllImageWeights,
            (int) imageIdx);

      (imageIdx < positiveImages) ?
      // if positive Image, add image weight
      // Sp += imageWeights[imageIdx] :
            Sp += imageWeight :
            // if negative Image, add image weight
            Sn += imageWeight;
   }

   return available;
}

__forceinline__ __device__ void getFeatureError(
      const FeatureValues & featureValues, const uint32_t positiveImages,
      // const double * const imageWeights,
      const int32_t threshold, const int32_t polarity, double & errorSum)
{
   errorSum = 0.0;
   const uint32_t featureValueCount = featureValues.getCount();
   const int32_t * const featureValueData = featureValues.getData();
   const uint32_t * const imageIdxs = featureValues.getSortedIdx();

   for (uint32_t i = 0; i < featureValueCount; ++i)
   {
      const uint32_t imageIdx = imageIdxs[i];
      const int32_t featureValue = featureValueData[i];
      //const double   imageWeight  = imageWeights[imageIdx];
      const double imageWeight = fetch_double(texAllImageWeights,
            (int) imageIdx);

      if (featureValue != INT_MAX)
      {
         const int8_t imageType = (imageIdx < positiveImages) ? 1 : 0;
         const int32_t h =
               ((polarity * featureValue) < (polarity * threshold)) ? 1 : 0;

         if ((h - imageType) != 0)
         {
            errorSum += imageWeight;
         }
      }
      else
      {
         errorSum += imageWeight;
      }
   }
}

__forceinline__ __device__ bool evaluateFeature(
      const int32_t * integralImages, const uint32_t availableImages[],
      const uint32_t imageWidth, const uint32_t imageHeight,
      const uint32_t countImages, const uint32_t positiveImages,
      // const double * const imageWeights,
      const double Tp, const double Tn, FeatureValues & featureValues,
      const uint8_t * const classifierData, const uint32_t classifierScaleIdx,
      const uint32_t classifierWidth, const uint32_t classifierHeight,
      const uint32_t x, const uint32_t y, double & error, int32_t & threshold,
      int32_t & polarity)
{
   bool success = false;

   success = getFeatureValues(
         integralImages,
         availableImages,
         countImages,
         imageWidth,
         imageHeight,
         classifierData,
         classifierScaleIdx,
         classifierWidth, classifierHeight,
         x, y,
         featureValues);

   if (!success)
   {
      error = 1.0;
      return false;
   }

   success = getThresholdAndPolarity(featureValues,
   // imageWeights,
         Tp, Tn, positiveImages, threshold, polarity);

   if (!success)
   {
      error = 1.0;
      return false;
   }

   getFeatureError(featureValues, positiveImages,
   //imageWeights,
         threshold, polarity, error);

   return true;
}

__global__ void getFeatureValuesGpu(
      const int32_t * integralImages,
      const uint32_t availableImages[],
      const uint32_t imageWidth,
      const uint32_t imageHeight,
      const uint32_t countImages,
      const uint32_t positiveImages, const uint8_t * const classifierData,
      const uint32_t x, const uint32_t y, const uint32_t classifierTypeIdx,
      const uint32_t classifierIdx, GetFeatureValueResult * featureValues)
{
   const uint32_t imageIdx = blockIdx.x * blockDim.x + threadIdx.x;

   if (countImages <= imageIdx)
      return;

   const int32_t * integralImage = integralImages + imageWidth * imageHeight * availableImages[imageIdx];

   // get all classifier of one type
   uint32_t classifierHeight;
   uint32_t classifierWidth;
   uint32_t classifierCount;
   const uint8_t * allClassifiers = NULL;
   const uint8_t * singleClassifier = NULL;

   Classifier::getClassifier(classifierData, classifierTypeIdx, classifierCount,
         &allClassifiers, classifierWidth, classifierHeight);
   assert(allClassifiers);

   uint32_t rectWidth;
   uint32_t rectHeight;
   Classifier::getClassifierScale(allClassifiers, classifierIdx,
         &singleClassifier, rectWidth, rectHeight);

   assert(singleClassifier);
#ifdef DEBUG
   if (imageIdx == 0)
   {
      printf(
            "Debug getFeatureValuesGpu classifierWidth:%d, classifierHeight:%d, rectWidth:%d, rectHeight:%d, classifierIdx:%d, classifierType:%d\n",
            classifierWidth, classifierHeight, rectWidth, rectHeight,
            classifierIdx, classifierTypeIdx);
   }
#endif

   Classifier::getFeatureValue(integralImage, singleClassifier, imageWidth, x, y, rectWidth,
         rectHeight, classifierWidth, classifierHeight,
         featureValues[imageIdx].featureValue);

   featureValues[imageIdx].imageIdx = imageIdx;
   featureValues[imageIdx].imageType = (positiveImages > imageIdx) ? 1 : 0;
}

__global__ void getBestClassifier(uint32_t countImages,
      int32_t * integralImages,
      const uint32_t availableImages[],
      const uint32_t positiveImages,
      const uint8_t * const classifierData, const uint32_t classifierDataSize,
      // const double * const imageWeights,
      const uint32_t imageWidth, const uint32_t imageHeight,
      const uint32_t ratioX, const uint32_t ratioY,
      Classifier::SelectionResult results[])
{
#ifdef DEBUG
//   if (blockIdx.x == 0 && threadIdx.x == 0)
//   {
//      printf("Debug: Kernel getBestClassifier Called block:%d thread:%d\n", blockIdx.x, threadIdx.x);
//   }

//   size_t mem_free_0, mem_tot_0;
//   cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
//   printf("Debug: Free memory: %d Mem total: %d\n", mem_free_0, mem_tot_0);

#endif
   const uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;

   const uint32_t imageWidthByRatio = imageWidth / ratioX;
   const uint32_t imageHeightByRatio = imageHeight / ratioY;

   if (pixel >= (imageWidthByRatio * imageHeightByRatio))
   {
      return;
   }

   const uint32_t yUnscaled = pixel / imageWidthByRatio;
   const uint32_t y = yUnscaled * ratioY;
   const uint32_t x = (pixel - yUnscaled * imageWidthByRatio) * ratioX;

   assert(x < imageWidth);
   assert(y < imageHeight);

   uint32_t classifierTypes;
   Classifier::getClassifierTypesCount(classifierData, classifierTypes);
   uint32_t bestClassifierType = 0;
   uint32_t bestClassifier = 0;
   double minError = DBL_MAX;
   int32_t threshold = INT32_MAX;
   int32_t polarity = INT32_MAX;

   double Tp = 0.0; // total image weight of positive images
   double Tn = 0.0; // total image weight of negative images

   for (uint32_t imageIdx = 0; imageIdx < positiveImages; ++imageIdx)
   {
      //Tp += imageWeights[imageIdx];
      Tp += fetch_double(texAllImageWeights, (int) imageIdx);
   }

   for (uint32_t imageIdx = positiveImages; imageIdx < countImages; ++imageIdx)
   {
      //Tn += imageWeights[imageIdx];
      Tn += fetch_double(texAllImageWeights, (int) imageIdx);
   }

   FeatureValues featureValues(countImages);

   for (uint32_t classifierTypeIdx = 0; classifierTypeIdx < classifierTypes;
         ++classifierTypeIdx)
   {
      // get all classifier of one type
      uint32_t featureHeight;
      uint32_t featureWidth;
      uint32_t classifierCount;
      const uint8_t * classifiers = NULL;

      Classifier::getClassifier(classifierData, classifierTypeIdx,
            classifierCount, &classifiers, featureWidth, featureHeight);
      assert(classifiers);

      // evaluate all scales
      for (uint32_t classifierIdx = 0; classifierIdx < classifierCount;
            ++classifierIdx)
      {
         double error;
         int32_t tmpThreshold;
         int32_t tmpPolarity;

         evaluateFeature(
               integralImages, availableImages,
               imageWidth, imageHeight,
               countImages,
               positiveImages,
               // imageWeights,
               Tp, Tn, featureValues, classifiers, classifierIdx, featureWidth,
               featureHeight, x, y, error, tmpThreshold, tmpPolarity);

         if (error < minError)
         {
            bestClassifierType = classifierTypeIdx;
            bestClassifier = classifierIdx;
            minError = error;
            threshold = tmpThreshold;
            polarity = tmpPolarity;
         }
      }
   }

   featureValues.clear();

   Classifier::SelectionResult & result = results[pixel];
   result.classifierIdx = bestClassifier;
   result.classifierTypeIdx = bestClassifierType;
   result.error = minError;
   result.x = x;
   result.y = y;
   result.threshold = threshold;
   result.polarity = polarity;
}

void evaluateClassifier(const uint32_t imageCount,
      const uint32_t positiveImages,
      const std::vector<Classifier::SelectionResult> & classifierSelectionResults,
      const std::vector<GetFeatureValueResult *> & featureValuesPtrs,
      const std::vector<double> & betas, double d,     // desired detection rate
      double Dprev, // previous detection rate
      double & Fi,  // current false positive rate
      double & Di,  // current detection rate
      std::vector<uint32_t> & falsePositiveIdx, double & stageThreshold)
{
   std::vector<double> alphas;
   alphas.reserve(betas.size());
   double thresholdUpperBound = 0.0;
   const double D = Dprev * d;

   for (std::vector<double>::const_iterator betaIter = betas.begin();
         betaIter != betas.end(); ++betaIter)
   {
      const double beta = *betaIter;
      double alpha = 40;
      if (beta != 0)
      {
         alpha = log(1.0 / beta);
      }

      alphas.push_back(alpha);
      thresholdUpperBound += alpha;
   }

   std::vector<uint8_t> strongClassifierResult;
   strongClassifierResult.reserve(imageCount);
   std::vector<double> sortedAlphas;

   // determine detection rate
   // only positive images
   for (uint32_t i = 0; i < positiveImages; ++i)
   {
      double sumAlphaH = 0.0;

      for (uint32_t k = 0; k < classifierSelectionResults.size(); ++k)
      {
         const Classifier::SelectionResult & classifierSelectionResult =
               classifierSelectionResults[k];
         const GetFeatureValueResult & featureValueResult =
               featureValuesPtrs[k][i];
         const int32_t featureValue = featureValueResult.featureValue;
         assert(featureValue != INT_MAX);

         const int32_t h =
               ((classifierSelectionResult.polarity * featureValue)
                     < (classifierSelectionResult.polarity
                           * classifierSelectionResult.threshold)) ? 1 : 0;

         sumAlphaH += alphas[k] * h;
      }

      sortedAlphas.push_back(sumAlphaH);
   }

   std::sort(sortedAlphas.rbegin(), sortedAlphas.rend());
   uint32_t alphaIdx = round(static_cast<double>(sortedAlphas.size()) * D);

   if (alphaIdx > 0)
   {
      alphaIdx--;
   }

   stageThreshold = sortedAlphas[alphaIdx];

   for (uint32_t i = 0; i < imageCount; ++i)
   {
      double sumAlphaH = 0.0;

      for (uint32_t k = 0; k < classifierSelectionResults.size(); ++k)
      {
         const Classifier::SelectionResult & classifierSelectionResult =
               classifierSelectionResults[k];
         const GetFeatureValueResult & featureValueResult =
               featureValuesPtrs[k][i];
         const int32_t featureValue = featureValueResult.featureValue;
         assert(featureValue != INT_MAX);

         const int32_t h =
               ((classifierSelectionResult.polarity * featureValue)
                     < (classifierSelectionResult.polarity
                           * classifierSelectionResult.threshold)) ? 1 : 0;

         sumAlphaH += alphas[k] * h;
      }

      if (sumAlphaH >= stageThreshold)
      {
         strongClassifierResult.push_back(1);
      }
      else
      {
         strongClassifierResult.push_back(0);
      }
   }

#ifdef DEBUG
   std::cout << "Debug: Sorted alpha index:" << alphaIdx << " Sorted alphas:";

   for (uint32_t i = 0; i < sortedAlphas.size(); ++i)
   {
      std::cout << sortedAlphas[i] << " ";
   }
   std::cout << std::endl;

   std::cout << "Debug: Cascade Threshold:" << stageThreshold << std::endl;
   std::cout << "Debug: Cascaded Classifier result for all images: ";
#endif

   uint32_t detectedObjects = 0;
   uint32_t falsePositives = 0;

   for (uint32_t i = 0; i < positiveImages; ++i)
   {
      if (strongClassifierResult[i] == 1)
      {
         detectedObjects++;
      }
   }

   falsePositiveIdx.clear();
   falsePositiveIdx.reserve(imageCount);

   for (uint32_t i = positiveImages; i < strongClassifierResult.size(); ++i)
   {
      if (strongClassifierResult[i] != 0)
      {
         falsePositives++;

         std::vector<uint32_t>::iterator falsePositiveIdxIter =
               std::upper_bound(falsePositiveIdx.begin(),
                     falsePositiveIdx.end(), i);

         if (falsePositiveIdxIter == falsePositiveIdx.end())
         {
            falsePositiveIdx.push_back(i);
         }
         else if (*falsePositiveIdxIter != i)
         {
            falsePositiveIdx.insert(falsePositiveIdxIter, i);
         }
      }
   }

   Di = static_cast<double>(detectedObjects) / positiveImages;
   Fi = static_cast<double>(falsePositives) / (imageCount - positiveImages);

#ifdef DEBUG
   std::cout << "Detected Objects: " << detectedObjects;
   std::cout << " False Positives: " << falsePositives;
   std::cout << " Di (rate): " << Di << " Fi (rate):" << Fi << std::endl;
#endif
}

void updateImageWeights(const uint32_t imageCount,
      const uint32_t positiveImages,
      const int32_t * integralImages,
      const uint32_t * availableImages,
      const uint32_t imageWidth,
      const uint32_t imageHeight,
      const FeatureTypes & featureTypes,
      const Classifier::SelectionResult & classifierSelectionResult,
      double * imageWeights, std::vector<double> & betas,
      std::vector<GetFeatureValueResult *> & featureValuesPtrs)
{
   size_t mem_tot_0 = 0;
   size_t mem_free_0 = 0;

   GetFeatureValueResult * featureValues = NULL;
   CUDA_CHECK_RETURN(
         cudaHostAlloc((void ** )&featureValues,
               imageCount * sizeof(GetFeatureValueResult), cudaHostAllocMapped));
   assert(featureValues);

   GetFeatureValueResult * featureValuesGpu = NULL;
   CUDA_CHECK_RETURN(
         cudaHostGetDevicePointer((void ** )&featureValuesGpu,
               (void * ) featureValues, 0));
   assert(featureValuesGpu);

   featureValuesPtrs.push_back(featureValues);

#ifdef DEBUG
   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << "Debug: getFeatureValuesGpu:" << std::endl;
   std::cout << "Debug: Free: " << mem_free_0 << " Mem total: " << mem_tot_0
         << std::endl;
#endif

   getFeatureValuesGpu<<<(imageCount + WORK_SIZE - 1) / WORK_SIZE, WORK_SIZE>>>(
         integralImages,
         availableImages,
         imageWidth,
         imageHeight,
         imageCount, positiveImages,
         //    featureTypes.getGpuData(),
         featureTypes.getConstantFeatureData(), classifierSelectionResult.x,
         classifierSelectionResult.y,
         classifierSelectionResult.classifierTypeIdx,
         classifierSelectionResult.classifierIdx, featureValuesGpu);

   CUDA_CHECK_RETURN(cudaPeekAtLastError());
   CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
   CUDA_CHECK_RETURN(cudaGetLastError());

   double beta;

   if (classifierSelectionResult.error != 1.0)
   {
      beta = classifierSelectionResult.error
            / (1.0 - classifierSelectionResult.error);
   }
   else
   {
      beta = DBL_MAX;
#ifdef DEBUG
      std::cout << "Debug: Classifier: Beta == Double Max" << std::endl;
#endif
   }
#ifdef DEBUG
   std::cout << "Debug: Classifier: "
         << Classifier::dumpSelectedClassifier(classifierSelectionResult,
               featureTypes) << std::endl;
#endif

   betas.push_back(beta);
   double sumWeigth = 0.0;

   if (beta != 0.0)
   {
#ifdef DEBUG
      std::cout << "Classifier Result (Polarity, Threshold) ["
            << classifierSelectionResult.polarity << ","
            << classifierSelectionResult.threshold << "]" << std::endl;
#endif

#ifdef DEBUG
      std::cout
            << "Classifier result for all images: (feature value, image type, error, h, image weight)";
#endif

      for (uint32_t i = 0; i < imageCount; ++i)
      {
         const GetFeatureValueResult & featureValueResult = featureValues[i];
         const int32_t featureValue = featureValueResult.featureValue;
         assert(featureValue != INT_MAX);
         const int32_t imageType = featureValueResult.imageType;

         const int32_t h =
               ((classifierSelectionResult.polarity * featureValue)
                     < (classifierSelectionResult.polarity
                           * classifierSelectionResult.threshold)) ? 1 : 0;
         int32_t error = 0;

         if (h != imageType)
         {
            error = 1;
         }

         double & weight = imageWeights[i];
#ifdef DEBUG
         std::cout << "[" << featureValue << "," << imageType << "," << error
               << "," << h << "," << weight << "],";

         if (((i + 1) % 10) == 0)
         {
            std::cout << std::endl;
         }
#endif
         weight = weight * pow(beta, 1 - error);
         sumWeigth += weight;
      }

#ifdef DEBUG
      std::cout << std::endl;
      std::cout << "Debug: Image Weights: " << std::endl;
#endif

      // normalize
      if (sumWeigth != 0.0)
      {
         for (uint32_t i = 0; i < imageCount; ++i)
         {
            double & imageWeight = imageWeights[i];
            imageWeight = imageWeight / sumWeigth;

#ifdef DEBUG
            std::cout << imageWeight;

            if (i + 1 < imageCount)
            {
               std::cout << ",";
            }
#endif
         }

         std::cout << std::endl;
      }
      else
      {
#ifdef DEBUG
         std::cout << " Sum of image weights is 0! ";
         for (uint32_t i = 0; i < imageCount; ++i)
         {
            std::cout << imageWeights[i];

            if (i + 1 < imageCount)
            {
               std::cout << ",";
            }
         }
#endif
      }
   }

   std::cout << "Beta:" << beta << std::endl;
}

void adaBoost(const uint32_t roundIdx, const uint32_t countImages,
      const uint32_t countPosImages, const uint32_t pixelCount,
      const uint32_t ratioX, const uint32_t ratioY, const uint32_t imageWidth,
      const uint32_t imageHeight, const FeatureTypes & featureTypes,
      int32_t * integralImages,
      const uint32_t availableImages[],
      double * imageWeightsPtr,
      double * gpuImageWeightsPtr, cudaEvent_t & start, cudaEvent_t & stop,
      std::vector<Classifier::SelectionResult> & allSelectedClassifier,
      Classifier::SelectionResult * results,
      Classifier::SelectionResult * gpuResults, uint32_t & newClassifierCount,
      std::ofstream & outputfile)
{
   size_t mem_tot_0 = 0;
   size_t mem_free_0 = 0;
#ifdef DEBUG
   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << "Debug: Get best classifier. Round:" << roundIdx << std::endl;
   std::cout << "Debug: Free memory: " << mem_free_0 << " Mem total: "
         << mem_tot_0 << std::endl;

   // register texture
   CUDA_CHECK_RETURN(
         cudaBindTexture( NULL, texAllImageWeights, gpuImageWeightsPtr, countImages * sizeof(double)));

   cudaEventRecord(start);
#endif
   getBestClassifier<<<(pixelCount + WORK_SIZE - 1) / WORK_SIZE, WORK_SIZE,
         featureTypes.getDataSize()>>>(countImages, integralImages,
         availableImages,
         countPosImages,
         //      featureTypes.getGpuData(),
         featureTypes.getConstantFeatureData(), featureTypes.getDataSize(),
         // gpuImageWeightsPtr, // stored in texture
         imageWidth, imageHeight, ratioX, ratioY, gpuResults);

   CUDA_CHECK_RETURN(cudaPeekAtLastError());
   CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
   CUDA_CHECK_RETURN(cudaGetLastError());

   // un-register texture
   CUDA_CHECK_RETURN(cudaUnbindTexture(texAllImageWeights));

#ifdef DEBUG
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   float elapsedTime = 0;
   cudaEventElapsedTime(&elapsedTime, start, stop);
   std::cout << "Debug: getBestClassifier Elapsed time: " << elapsedTime / 1000
         << "s" << std::endl;
   std::cout << "Classifier Round:" << roundIdx << " Done!" << std::endl;
   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << "Debug: Free memory: " << mem_free_0 << " Mem total: "
         << mem_tot_0 << std::endl;

   std::cout << "First classifiers:" << std::endl;

   for (uint32_t i = 0; i < pixelCount && i < 10; ++i)
   {
      const Classifier::SelectionResult & selectedClassifier = results[i];
      std::cout << "[" << selectedClassifier.classifierTypeIdx << ","
            << selectedClassifier.classifierIdx << ","
            << selectedClassifier.error << "," << selectedClassifier.polarity
            << "," << selectedClassifier.threshold << ","
            << selectedClassifier.x << "," << selectedClassifier.y << "],";
   }

   if (pixelCount > 10)
   {
      std::cout << "...,";

      for (uint32_t i = pixelCount - 10; i < pixelCount || i == 25; ++i)
      {
         const Classifier::SelectionResult & selectedClassifier = results[i];
         std::cout << "[" << selectedClassifier.classifierTypeIdx << ","
               << selectedClassifier.classifierIdx << ","
               << selectedClassifier.error << "," << selectedClassifier.polarity
               << "," << selectedClassifier.threshold << ","
               << selectedClassifier.x << "," << selectedClassifier.y << "]";

         if ((pixelCount + 1) < pixelCount)
         {
            std::cout << ",";
         }
      }
   }

   std::cout << std::endl;
#endif

   Classifier::SelectionResult selectedClassifier;
   selectedClassifier.error = DBL_MAX;
   newClassifierCount = 0;

   // select best classifier, but don't select one we already have
   for (uint32_t i = 0; i < pixelCount; ++i)
   {
      const Classifier::SelectionResult & tmpSelectedClassifier = results[i];

      bool isAlreadySelected = false;

      for (uint32_t storedClassifierIdx = 0;
            storedClassifierIdx < allSelectedClassifier.size();
            ++storedClassifierIdx)
      {
         if (tmpSelectedClassifier == allSelectedClassifier[storedClassifierIdx]
               && tmpSelectedClassifier.error == 0.0)
         {
            isAlreadySelected = true;
            break;
         }
      }

      if (isAlreadySelected)
      {
         continue;
      }

      if (tmpSelectedClassifier.error == 0.0)
      {
         // select all error free classifiers
         allSelectedClassifier.push_back(tmpSelectedClassifier);
         newClassifierCount++;
      }
      else if (selectedClassifier.error > tmpSelectedClassifier.error)
      {
         selectedClassifier = tmpSelectedClassifier;
      }
   }

   if (selectedClassifier.error < DBL_MAX)
   {
      allSelectedClassifier.push_back(selectedClassifier);
      newClassifierCount++;
   }

   for (uint32_t i = 0; i < newClassifierCount; ++i)
   {
      std::string prettyClassifier = Classifier::dumpSelectedClassifier(
            allSelectedClassifier[allSelectedClassifier.size() - i - 1],
            featureTypes);
      outputfile << prettyClassifier << std::endl;
      std::cout << prettyClassifier << std::endl;
   }

   outputfile.flush();
}

#endif /* LEARNER_H_ */
