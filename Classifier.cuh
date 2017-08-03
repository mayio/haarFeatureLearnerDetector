/*
 *
 *  Created on: May 17, 2017
 *      Author: Mario Lüder
 *
 */


#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include "FeatureTypes.cuh"

#include <vector>
#include <string>
#include <sstream>

#include "stdint.h"
#include "assert.h"
#include "stdio.h"

//#include "opencv2/core/cuda_devptrs.hpp"
#include <opencv2/cudev/ptr2d/gpumat.hpp>



// forward declaration
namespace cv {
   namespace gpu {
      class GpuMat;
   }
}

class Classifier
{
public:
   Classifier();
   virtual ~Classifier();

   struct SelectionResult
   {
      uint32_t classifierTypeIdx;
      uint32_t classifierIdx;
      double error;
      uint32_t x;
      uint32_t y;
      int32_t threshold;
      int32_t polarity;

      bool operator==(const SelectionResult & r) const
      {
         if (r.classifierTypeIdx == classifierTypeIdx
               && r.classifierIdx == classifierIdx && r.x == x && r.y == y
               && r.error == error)
         {
            return true;
         }

         return false;
      }
   };

   struct Stage
   {
      std::vector<double> betas;
      std::vector<Classifier::SelectionResult> stagedClassifier;
      double stageThreshold;
   };

   struct ClassificationResult
   {
      uint32_t x;
      uint32_t y;
      uint32_t width;
      uint32_t height;
      double strength;
   };

   __forceinline__ __host__ __device__ static void getClassifierTypesCount(
         const uint8_t * const classifierData, uint32_t & count)
   {
      assert(classifierData);
      count = *(uint32_t *) (classifierData);
   }

   __forceinline__ __host__ __device__ static void getClassifier(const uint8_t * const classifierData,
         const uint32_t classifierTypeIdx, uint32_t & count, const uint8_t ** classifierTypeData,
         uint32_t & width, uint32_t & height)
   {
      assert(classifierData);
      const uint32_t offset = ((uint32_t *) (classifierData))[classifierTypeIdx + 1];
      (*classifierTypeData) = classifierData + offset;
      width = ((uint32_t *) (*classifierTypeData))[0];
      height = ((uint32_t *) (*classifierTypeData))[1];
      count = ((uint32_t *) (*classifierTypeData))[2];
   }

   __forceinline__ __host__ __device__ static void getClassifierScale(
         const uint8_t * const classifierData, const uint32_t classifierScaleIdx,
         const uint8_t ** scaledClassifierData, uint32_t & rectWidth, uint32_t & rectHeight)
   {
      assert(classifierData);
      const uint32_t classifierWidth = ((uint32_t *) (classifierData))[0];
      const uint32_t classifierHeight = ((uint32_t *) (classifierData))[1];
      const uint32_t rectangleSize = 3 * sizeof(uint32_t);
      const uint32_t headerSize = 3 * sizeof(uint32_t);
      const uint32_t classifierSize = classifierWidth * classifierHeight
            * rectangleSize;
      const uint32_t offset = classifierSize * classifierScaleIdx + headerSize;
      (*scaledClassifierData) = classifierData + offset;
      rectWidth = ((uint32_t *) (*scaledClassifierData))[0];
      rectHeight = ((uint32_t *) (*scaledClassifierData))[1];
   }

   __forceinline__ __host__ __device__ static void getRectangleType(const uint8_t * const classifierData,
         const uint32_t rectIdx, int32_t & type)
   {
      assert(classifierData);
      const uint32_t rectangleSize = 3 * sizeof(uint32_t);
      type = ((int32_t *) (classifierData + rectangleSize * rectIdx))[2];
   }

/*
   __forceinline__ __device__ static void getFeatureValue(const uint8_t * const classifier,
         const int32_t * integralImage, const uint32_t x,
         const uint32_t y, const uint32_t rectElementWidth,
         const uint32_t rectElementHeight, const uint32_t featureWidth,
         const uint32_t featureHeight, int32_t & value)
   {
      int32_t * integralImageLine = (int32_t*)((uint8_t*)(integralImage.data) + y * integralImage.step);
      getFeatureValue(
            classifier,
            integralImageLine,
            integralImage.step,
            x,
            rectElementWidth,
            rectElementHeight,
            featureWidth,
            featureHeight,
            value
            );
   }
*/
   __forceinline__ __device__ static void getFeatureValue(
         const int32_t * integralImage,
         const uint8_t * const classifier,
         const uint32_t lineStep,
         const uint32_t x,
         const uint32_t y,
         const uint32_t rectElementWidth, const uint32_t rectElementHeight,
         const uint32_t featureWidth, const uint32_t featureHeight,
         int32_t & value)
   {
// FIXME remove this
//printf("x:%d y:%d lineStep:%d rectWidth:%d rectHeight:%d featureWidth:%d featureHeight:%d\n",
//      x,y,lineStep,rectElementWidth,rectElementHeight,featureWidth,featureHeight);


      value = 0;
      const uint32_t rectElementWidthErode = rectElementWidth - 1;
      const uint32_t rectElementHeightErode = rectElementHeight -1;

      uint32_t xi = x;

      const int32_t * integralImageLine  = integralImage + y * lineStep;
      const int32_t * integralImageLine2 = integralImageLine + rectElementHeightErode * lineStep;

// FIXME remove this
//printf("integralImageLine:%#010X integralImageLine2:%#010X \n", integralImageLine, integralImageLine2);

      int32_t i1 = integralImageLine[x];                          // integralImage(yi, xi);
      int32_t i2 = integralImageLine[x + rectElementWidthErode];  // integralImage(yi, xi + rectElementWidth);
      int32_t i3 = integralImageLine2[x];                         //integralImage(yi + rectElementHeight, xi);
      int32_t i4 = integralImageLine2[x + rectElementWidthErode]; //integralImage(yi + rectElementHeight, xi + rectElementWidth);

      int32_t k3 = i3;
      int32_t k4 = i4;

      for (uint32_t h = 0; h < featureHeight; ++h)
      {
         for (uint32_t w = 0; w < featureWidth; ++w)
         {
            int32_t rectangleType = 0;
            Classifier::getRectangleType(classifier, h * featureWidth + w, rectangleType);
            value += ((i4 + i1 - i2 - i3) * rectangleType);

// FIXME remove this
//printf("h:%d w:%d xi:%d i1:%d i2:%d i3:%d i4:%d val:%d rectType:%d\n",h,w,xi,i1,i2,i3,i4,value,rectangleType);

            if (w + 1 < featureWidth)
            {
// FIXME remove this
//printf(" --- right shift --- \n");
               i1 = i2;
               i3 = i4;

               xi += rectElementWidthErode;
               const uint32_t xi2 = xi + rectElementWidthErode;
               i2 = integralImageLine[xi2];
               i4 = integralImageLine2[xi2];
            }
         }

         if (h + 1 < featureHeight)
         {
// FIXME remove this
//printf(" --- bottom shift --- \n");
            xi = x;
            integralImageLine = integralImageLine2;
            integralImageLine2 = integralImageLine2 + rectElementHeightErode * lineStep;
// FIXME remove this
//printf("integralImageLine:%#010X integralImageLine2:%#010X \n", integralImageLine, integralImageLine2);
            i1 = k3;
            i2 = k4;
            i3 = integralImageLine2[xi];
            i4 = integralImageLine2[xi + rectElementWidthErode];

            k3 = i3;
            k4 = i4;
         }
      }
   }

   __forceinline__ __device__ static void getFeatureValueTex(
         const cudaTextureObject_t & texIntegralImage,
         const uint8_t * const classifier,
         const uint32_t x,
         uint32_t y,
         const uint32_t rectElementWidth, const uint32_t rectElementHeight,
         const uint32_t featureWidth, const uint32_t featureHeight,
         int32_t & value)
   {
      value = 0;
      uint32_t xi = x;

      const uint32_t rectElementWidthErode = rectElementWidth - 1;
      const uint32_t rectElementHeightErode = rectElementHeight -1;

      uint32_t y2 = y + rectElementHeightErode;
      uint32_t x2 = x + rectElementWidthErode;

      int32_t i1 = tex2D<int32_t>(texIntegralImage, x,  y); //  integralImageLine[x];                     // integralImage(yi, xi);
      int32_t i2 = tex2D<int32_t>(texIntegralImage, x2, y); //  integralImageLine[x + rectElementWidth];  // integralImage(yi, xi + rectElementWidth);
      int32_t i3 = tex2D<int32_t>(texIntegralImage, x,  y2); // integralImageLine2[x];                    //integralImage(yi + rectElementHeight, xi);
      int32_t i4 = tex2D<int32_t>(texIntegralImage, x2, y2); //integralImageLine2[x + rectElementWidth]; //integralImage(yi + rectElementHeight, xi + rectElementWidth);

      int32_t k3 = i3;
      int32_t k4 = i4;

      for (uint32_t h = 0; h < featureHeight; ++h)
      {
         for (uint32_t w = 0; w < featureWidth; ++w)
         {
            int32_t rectangleType = 0;
            Classifier::getRectangleType(classifier, h * featureWidth + w, rectangleType);
            value += ((i4 + i1 - i2 - i3) * rectangleType);

            if (w + 1 < featureWidth)
            {
               i1 = i2;
               i3 = i4;

               xi += rectElementWidthErode;
               const uint32_t xi2 = xi + rectElementWidthErode;
               i2 = tex2D<int32_t>(texIntegralImage, xi2, y); // integralImageLine[xi2];
               i4 = tex2D<int32_t>(texIntegralImage, xi2, y2); // integralImageLine2[xi2];
            }
         }

         if (h + 1 < featureHeight)
         {
            xi = x;
            y = y2; // integralImageLine = integralImageLine2;
            y2 += rectElementHeightErode; // integralImageLine2 = (int32_t*)((uint8_t*)(integralImageLine2) + rectElementHeight * lineStep);
            i1 = k3;
            i2 = k4;
            i3 = tex2D<int32_t>(texIntegralImage, xi, y2); // integralImageLine2[xi];
            i4 = tex2D<int32_t>(texIntegralImage, xi + rectElementWidthErode, y2); // integralImageLine2[xi + rectElementWidth];

            k3 = i3;
            k4 = i4;
         }
      }
   }

   static void sizeStrongClassifier(
         const std::vector<Classifier::Stage> & strongClassifier,
         const FeatureTypes & featureTypes,
         uint32_t & xMin,
         uint32_t & yMin,
         uint32_t & xMax,
         uint32_t & yMax);


   static void scaleStrongClassifier(
         const double scale,
         const std::vector<Classifier::Stage> & strongClassifier,
         const FeatureTypes & featureTypes,
         std::vector<Classifier::Stage> & scaledStrongClassifier,
         FeatureTypes & scaledFeatureTypes);

   static bool fromResult(const std::string & result, std::vector<Classifier::Stage> & strongClassifier, FeatureTypes & featureTypes);

   static std::string dumpSelectedClassifier(
         const Classifier::SelectionResult & selected, const FeatureTypes & featureTypes);

   static bool detectStrongClassifier(
         const std::vector<Classifier::Stage> & strongClassifier,
         FeatureTypes & featureTypes,
         const cv::cuda::GpuMat & gpuIntegralImage,
         std::vector<Classifier::ClassificationResult> & results
         );

   static void detectStrongClassifierOnImageSet(
         const std::vector<Classifier::Stage> & strongClassifier,
         FeatureTypes & featureTypes,
         const int32_t * const gpuIntegralImages,
         const uint32_t startImageIdx,
         const uint32_t imageCount,
         const uint32_t imageWidth,
         const uint32_t imageHeight,
         bool * results
         );

   static texture<int32_t, 2> & getTexIntegralImage();
private:
   template<typename T>
   static bool parseValue(const std::string & str, const char * const delimiter, const char * const expected, std::string::size_type & pos, std::string::size_type & lastPos, T & value)
   {
      if (pos == std::string::npos)
      {
         return false;
      }
      pos++;
      lastPos = pos;
      pos = str.find_first_of(delimiter,pos);

      bool hasExpectedDelimiter = false;

      for (uint32_t i = 0; ((expected[i] != '\0') && (hasExpectedDelimiter == false)); ++i)
      {
         if (expected[i] == str[pos])
         {
            hasExpectedDelimiter = true;
            break;
         }
      }

      if ((pos != std::string::npos) && hasExpectedDelimiter)
      {
         std::stringstream ss(str.substr (lastPos,pos - lastPos));
         ss >> value;
         return true;
      }

      pos = std::string::npos;
      lastPos = std::string::npos;
      return false;
   }

   static inline void addUniqueResult(const Classifier::ClassificationResult & newResult, std::vector<Classifier::ClassificationResult> & results);
};

#endif /* CLASSIFIER_H_ */
