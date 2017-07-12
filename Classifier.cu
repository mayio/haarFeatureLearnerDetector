/*
 *
 *  Created on: May 17, 2017
 *      Author: Mario Lüder
 *
 *
 */

#include "Classifier.cuh"
#include "defines.cuh"
#include "GpuStrongClassifier.cuh"
#include "utilities.cuh"

#include <opencv2/core/gpumat.hpp>

#include <iostream>

// texture
texture<int32_t, 2, cudaReadModeElementType> texIntegralImage;

__forceinline__ __device__ void Classifier::getFeatureValueTex(
      const uint8_t * const classifier,
      const uint32_t x,
      uint32_t y,
      const uint32_t rectElementWidth, const uint32_t rectElementHeight,
      const uint32_t featureWidth, const uint32_t featureHeight,
      int32_t & value)
{
   assert(false);

   // FIXME:
   // this is not right
   // see getFeatureValue
   // the rectElementWidth and rectElementHeight have to be reduced by 1
   //

   value = 0;
   uint32_t xi = x;

   uint32_t y2 = y + rectElementHeight;
   uint32_t x2 = x + rectElementWidth;

   int32_t i1 = tex2D(texIntegralImage, x,  y); //  integralImageLine[x];                     // integralImage(yi, xi);
   int32_t i2 = tex2D(texIntegralImage, x2, y); //  integralImageLine[x + rectElementWidth];  // integralImage(yi, xi + rectElementWidth);
   int32_t i3 = tex2D(texIntegralImage, x,  y2); // integralImageLine2[x];                    //integralImage(yi + rectElementHeight, xi);
   int32_t i4 = tex2D(texIntegralImage, x2, y2); //integralImageLine2[x + rectElementWidth]; //integralImage(yi + rectElementHeight, xi + rectElementWidth);

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

            xi += rectElementWidth;
            const uint32_t xi2 = xi + rectElementWidth;
            i2 = tex2D(texIntegralImage, xi2, y); // integralImageLine[xi2];
            i4 = tex2D(texIntegralImage, xi2, y2); // integralImageLine2[xi2];
         }
      }

      if (h + 1 < featureHeight)
      {
         xi = x;
         y = y2; // integralImageLine = integralImageLine2;
         y2 += rectElementHeight; // integralImageLine2 = (int32_t*)((uint8_t*)(integralImageLine2) + rectElementHeight * lineStep);
         i1 = k3;
         i2 = k4;
         i3 = tex2D(texIntegralImage, xi, y2); // integralImageLine2[xi];
         i4 = tex2D(texIntegralImage, xi + rectElementWidth, y2); // integralImageLine2[xi + rectElementWidth];

         k3 = i3;
         k4 = i4;
      }
   }
}

std::string Classifier::dumpSelectedClassifier(
      const Classifier::SelectionResult & selected,
      const FeatureTypes & featureTypes)
{
   std::stringstream prettyClassifier;
   uint32_t maxFeatureTypes;

   Classifier::getClassifierTypesCount(featureTypes.getData(), maxFeatureTypes);
   assert(maxFeatureTypes > selected.classifierTypeIdx);

   uint32_t featureWidth;
   uint32_t featureHeight;
   uint32_t maxClassifier;
   const uint8_t * allClassifier = NULL;
   const uint8_t * selectedClassifierPtr = NULL;

   Classifier::getClassifier(featureTypes.getData(), selected.classifierTypeIdx, maxClassifier, &allClassifier,
         featureWidth, featureHeight);
   assert(maxClassifier > selected.classifierIdx);
   assert(allClassifier);

   uint32_t rectWidth;
   uint32_t rectHeight;

   Classifier::getClassifierScale(allClassifier, selected.classifierIdx, &selectedClassifierPtr,
         rectWidth, rectHeight);
   assert(selectedClassifierPtr);

   prettyClassifier << "[";
   prettyClassifier << "[";
   for (uint32_t i = 0; i < featureHeight; ++i)
   {
      prettyClassifier << "[";

      for (uint32_t j = 0; j < featureWidth; ++j)
      {
         prettyClassifier << "[";
         prettyClassifier << rectWidth << ",";
         prettyClassifier << rectHeight << ",";
         int32_t type;
         Classifier::getRectangleType(selectedClassifierPtr, i * featureWidth + j, type);
         prettyClassifier << type;
         prettyClassifier << "]";

         if ((j + 1) < featureWidth)
         {
            prettyClassifier << ",";
         }
      }

      prettyClassifier << "]";

      if ((i + 1) < featureHeight)
      {
         prettyClassifier << ",";
      }
   }
   prettyClassifier << "]";
   prettyClassifier << "," << selected.x;
   prettyClassifier << "," << selected.y;
   prettyClassifier << "," << selected.error;
   prettyClassifier << "," << selected.threshold;
   prettyClassifier << "," << selected.polarity;
   prettyClassifier << "]";

   return prettyClassifier.str();
}

bool Classifier::fromResult(const std::string & result, std::vector<Classifier::Stage> & strongClassifier, FeatureTypes & featureTypes)
{
   // [[ [[[[[49,22,1]],
   //    [[[[[
   const size_t strLength = result.size();

   if (strLength == 0)
   {
      return false;
   }

   std::string::size_type pos = result.find_first_of("[");
   std::string::size_type lastPos = pos;

   if (pos == std::string::npos)
   {
      return false;
   }

   pos++;
   lastPos++;
   const char * const delimiter = "[],";
   std::cout << "[";

   pos = result.find_first_of(delimiter, pos);
   while(pos != std::string::npos && result[pos] != ']')
   {
      // stage
      Classifier::Stage stage;

      if (pos != std::string::npos && result[pos] == ',')
      {
         pos++;
         lastPos = pos;
         pos = result.find_first_of(delimiter,pos);
         std::cout << ",";
      }

      if (pos != std::string::npos)
      {
         pos++;
         pos = result.find_first_of(delimiter, pos);
      }

      if (pos != std::string::npos && result[pos] == '[')
      {
         pos++;
         lastPos = pos;
         pos = result.find_first_of(delimiter,pos);
         std::cout << "[";
      }
      else
      {
         std::cout << " Unexpected delimiter. Expected ','" << std::endl;
         return false;
      }

      // [[[[[19,16,1]],[[19,16,-1]]],12,6,0,-12641,1],
      while(pos != std::string::npos && result[pos] != ']')
      {
         if (pos != std::string::npos && result[pos] == ',')
         {
            pos++;
            lastPos = pos;
            pos = result.find_first_of(delimiter,pos);
            std::cout << ",";
         }

         if (pos != std::string::npos && result[pos] == '[')
         {
            // Classifier with parameter
            std::cout << "[";
            pos++;
            pos = result.find_first_of(delimiter,pos);

            uint32_t width = 0;
            uint32_t height = 0;

            FeatureType feature(0,0);
            Classifier::SelectionResult selectionResult;
            selectionResult.classifierIdx = 0;
            selectionResult.classifierTypeIdx = featureTypes.size();

            if (pos != std::string::npos && result[pos] == '[')
            {
               // Classifier
               //[[[19,16,1]],[[19,16,-1]]]
               std::cout << "[";
               pos++;
               pos = result.find_first_of(delimiter,pos);

               while(pos != std::string::npos && result[pos] != ']')
               {
                  if (pos != std::string::npos && result[pos] == ',')
                  {
                     pos++;
                     lastPos = pos;
                     pos = result.find_first_of(delimiter,pos);
                     std::cout << ",";
                  }

                  if (pos != std::string::npos && result[pos] == '[')
                  {
                     // Row
                     feature.addRow();
                     std::cout << "[";
                     pos++;
                     pos = result.find_first_of(delimiter,pos);

                     while(pos != std::string::npos && result[pos] != ']')
                     {
                        if (pos != std::string::npos && result[pos] == ',')
                        {
                           pos++;
                           lastPos = pos;
                           pos = result.find_first_of(delimiter,pos);
                           std::cout << ",";
                        }

                        if (pos != std::string::npos && result[pos] == '[')
                        {
                           // Column
                           std::cout << "[";

                           if (parseValue(result, delimiter, ",", pos, lastPos, width))
                           {
                              std::cout << width << ",";
                           }

                           if (parseValue(result, delimiter, ",", pos, lastPos, height))
                           {
                              std::cout << height << ",";
                           }

                           int32_t type;

                           if (parseValue(result, delimiter, "]", pos, lastPos, type))
                           {
                              std::cout << type << "]";
                           }

                           feature << type;
                        }

                        if (pos != std::string::npos)
                        {
                           pos++;
                           lastPos = pos;
                           pos = result.find_first_of(delimiter,pos);
                        }
                     }
                     std::cout << "]";
                  }

                  if (pos != std::string::npos)
                  {
                     pos++;
                     lastPos = pos;
                     pos = result.find_first_of(delimiter,pos);
                  }
               }

               if (pos != std::string::npos)
               {
                  pos++;
                  lastPos = pos;
                  pos = result.find_first_of(delimiter,pos);
               }

               if (parseValue(result, delimiter, ",", pos, lastPos, selectionResult.x))
               {
                   std::cout << "]," << selectionResult.x << ",";
               }

               if (parseValue(result, delimiter, ",", pos, lastPos, selectionResult.y))
               {
                   std::cout << selectionResult.y << ",";
               }

               if (parseValue(result, delimiter, ",", pos, lastPos, selectionResult.error))
               {
                   std::cout << selectionResult.error << ",";
               }

               if (parseValue(result, delimiter, ",", pos, lastPos, selectionResult.threshold))
               {
                   std::cout << selectionResult.threshold << ",";
               }

               if (parseValue(result, delimiter, "]", pos, lastPos, selectionResult.polarity))
               {
                   std::cout << selectionResult.polarity << "]";
               }
            }

            feature.setRect(width, height);
            featureTypes.push_back(feature);
            stage.stagedClassifier.push_back(selectionResult);
         }

         if (pos != std::string::npos)
         {
            pos++;
            lastPos = pos;
            pos = result.find_first_of(delimiter,pos);
         }
      }

      // ,[0.0343905,0.110103],2.20634]
      //
      // betas
      //
      if (pos != std::string::npos)
      {
         std::cout << result[pos];
         pos++;
         lastPos = pos;
         pos = result.find_first_of(delimiter,pos);
      }

      if (pos != std::string::npos && result[pos] == ',')
      {
         pos++;
         lastPos = pos;
         pos = result.find_first_of(delimiter,pos);
         std::cout << ",";
      }
      else
      {
         std::cout << " Unexpected delimiter. Expected ','" << std::endl;
         return false;
      }

      if (pos != std::string::npos)
      {
         std::cout << result[pos];
      }

      while(pos != std::string::npos && result[pos] != ']')
      {
         double beta;

         if (parseValue(result, delimiter, ",]", pos, lastPos, beta))
         {
            stage.betas.push_back(beta);
            std::cout << beta << result[pos];
         }
      }

      // stage threshold

      if (pos != std::string::npos)
      {
         pos++;
         lastPos = pos;
         pos = result.find_first_of(delimiter,pos);
      }

      if (pos != std::string::npos && result[pos] == ',')
      {
         std::cout << ",";
      }
      else
      {
         std::cout << " Unexpected delimiter. Expected ','" << std::endl;
         return false;
      }

      if (parseValue(result, delimiter, "]", pos, lastPos, stage.stageThreshold))
      {
         std::cout << stage.stageThreshold << result[pos];
      }

      strongClassifier.push_back(stage);

      if (pos != std::string::npos)
      {
         pos++;
         lastPos = pos;
         pos = result.find_first_of(delimiter,pos);
      }
      // stage end
      std::cout << std::endl;
   }

   std::cout << "]" << std::endl;
   return true;
}

void Classifier::sizeStrongClassifier(
         const std::vector<Classifier::Stage> & strongClassifier,
         const FeatureTypes & featureTypes,
         uint32_t & xMin,
         uint32_t & yMin,
         uint32_t & xMax,
         uint32_t & yMax)
{

   xMax = 0;
   yMax = 0;
   xMin = INT32_MAX;
   yMin = INT32_MAX;

   for (std::vector<Classifier::Stage>::const_iterator stageIter = strongClassifier.begin();
        stageIter != strongClassifier.end();
        ++stageIter)
   {
      const Classifier::Stage & stage = *stageIter;

      for (std::vector<Classifier::SelectionResult>::const_iterator selectionResultIter = stage.stagedClassifier.begin();
            selectionResultIter != stage.stagedClassifier.end();
            ++selectionResultIter)
      {
         const Classifier::SelectionResult & selectionResult = *selectionResultIter;
         xMin = (selectionResult.x < xMin) ? selectionResult.x : xMin;
         yMin = (selectionResult.y < yMin) ? selectionResult.y : yMin;

         const FeatureType & featureType = featureTypes[selectionResult.classifierTypeIdx];
         const uint32_t xRight  = selectionResult.x + (featureType.mFeatureWidth  * featureType.mRect.width);
         const uint32_t yBottom = selectionResult.y + (featureType.mFeatureHeight * featureType.mRect.height);

         xMax = (xRight  > xMax) ? xRight  : xMax;
         yMax = (yBottom > yMax) ? yBottom : yMax;
      }
   }

   printf("sizeStrongClassifier: xMin:%d, yMin:%d, xMax:%d, yMax:%d\n", xMin, yMin, xMax, yMax);
}

void Classifier::scaleStrongClassifier(
         const double scale,
         const std::vector<Classifier::Stage> & strongClassifier,
         const FeatureTypes & featureTypes,
         std::vector<Classifier::Stage> & scaledStrongClassifier,
         FeatureTypes & scaledFeatureTypes)
{
   scaledStrongClassifier = strongClassifier;
   scaledFeatureTypes = featureTypes;
   assert(scaledStrongClassifier.size() == strongClassifier.size());

   for (std::vector<Classifier::Stage>::iterator stageIter = scaledStrongClassifier.begin();
        stageIter != scaledStrongClassifier.end();
        ++stageIter)
   {
      Classifier::Stage & stage = *stageIter;

      for (std::vector<Classifier::SelectionResult>::iterator selectionResultIter = stage.stagedClassifier.begin();
            selectionResultIter != stage.stagedClassifier.end();
            ++selectionResultIter)
      {
         Classifier::SelectionResult & selectionResult = *selectionResultIter;
         selectionResult.x         = static_cast<uint32_t>(static_cast<double>(selectionResult.x) * scale);
         selectionResult.y         = static_cast<uint32_t>(static_cast<double>(selectionResult.y) * scale);
         selectionResult.threshold = static_cast<int32_t>(static_cast<double>(selectionResult.threshold) * pow(scale, 2));
      }
   }

   for (std::vector<FeatureType>::iterator featureTypeIter = scaledFeatureTypes.begin();
        featureTypeIter != scaledFeatureTypes.end();
        ++featureTypeIter)
   {
      FeatureType & featureType = *featureTypeIter;
      featureType.setRect(
            static_cast<uint32_t>(static_cast<double>(featureType.mRect.width) * scale),
            static_cast<uint32_t>(static_cast<double>(featureType.mRect.height) * scale),
            featureType.mRect.type);
   }
}

__device__ __forceinline__ void detectStrongClassifierAtPoint(
      const int32_t * const integralImage,
      const uint32_t imageWidth,
      const uint32_t imageHeight,
      const uint32_t step,
      const uint32_t x,
      const uint32_t y,
      const uint8_t * const allClassifierData,
      const GpuStrongClassifier::Stage * const stages,
      const uint32_t stageCount,
      bool & detected,
      double & hSum
      )
{
   detected = false;

   // for each stage
   for (uint32_t stageIdx = 0; stageIdx < stageCount; ++stageIdx)
   {
      hSum = 0.0;
      const GpuStrongClassifier::Stage & stage = stages[stageIdx];

      // for each classifier in stage
      for (uint32_t classifierIdx = 0;  classifierIdx < stage.mClassifierCount; ++classifierIdx)
      {
         const Classifier::SelectionResult & classifierDescription = stage.mSelectionResults[classifierIdx];
         const double beta = stage.mBetas[classifierIdx];

         double alpha = 40.0;

         if  (beta != 0.0)
             alpha = log(1.0/beta);

         // get all classifier of one type - here we have only one
         uint32_t featureHeight;
         uint32_t featureWidth;
         uint32_t classifierCount;
         const uint8_t * classifiers = NULL;

         Classifier::getClassifier(
               allClassifierData,
               classifierDescription.classifierTypeIdx,
               classifierCount,
               &classifiers,
               featureWidth,
               featureHeight);

         assert(classifiers);

         uint32_t rectWidth;
         uint32_t rectHeight;
         const uint8_t * singleClassifier = NULL;

         Classifier::getClassifierScale(
               classifiers,
               classifierDescription.classifierIdx,
               &singleClassifier,
               rectWidth,
               rectHeight);

         assert(singleClassifier);

         const uint32_t classifierLeftPoint = x + classifierDescription.x;
         const uint32_t classifierUpperPoint = y + classifierDescription.y;
         const uint32_t classifierRightPoint = classifierLeftPoint + rectWidth * featureWidth;
         const uint32_t classifierBottomPoint = classifierUpperPoint + rectHeight * featureHeight;

         const bool outOfRange = ((classifierRightPoint <= imageWidth)
               && (classifierBottomPoint <= imageHeight)) ? false : true;

         int32_t featureValue = INT_MAX;

/*TODO: debug message
         if (threadIdx.x == 0 && blockIdx.x == 0)
         {
            printf("stage:%d featureWidth:%d, featureHeight:%d, rectWidth:%d, rechtHeight:%d typeIdx:%d outOfRange:%d\n",
                  stageIdx, featureWidth, featureHeight, rectWidth, rectHeight, classifierDescription.classifierTypeIdx, outOfRange);
            printf("classifierLeftPoint:%d classifierUpperPoint:%d classifierRightPoint:%d classifierBottomPoint:%d\n",
                  classifierLeftPoint, classifierUpperPoint, classifierRightPoint, classifierBottomPoint);
         }
*/

         if (!outOfRange)
         {
            Classifier::getFeatureValue(
                  integralImage,
                  singleClassifier,
                  // FIXME check this
                  step,
                  classifierLeftPoint,
                  classifierUpperPoint,
                  rectWidth, rectHeight,
                  featureWidth, featureHeight,
                  featureValue);
            const int32_t h = (classifierDescription.polarity * featureValue) < (classifierDescription.polarity * classifierDescription.threshold) ? 1 : 0;
            hSum += static_cast<double>(h) * alpha;
// FIXME remove this
/*TODO: debug message
if (threadIdx.x == 1 && blockIdx.x == 1)
{
  printf("h(%d) = pol(%d) * val(%d) < pol(%d) * threshold(%d)\n", h, classifierDescription.polarity, featureValue, classifierDescription.polarity, classifierDescription.threshold);
  printf("hSum(%f) += h(%d) * alpha(%f)\n\n", hSum, h, alpha);
}
*/
            //alphaSum += alpha;
         }
         else
         {
            hSum = 0.0;
            detected = false;
            return;
         }
      }

      if (hSum < stage.mStageThreshold)
      {
         hSum = 0.0;
         detected = false;
         return;
      }
      else
      {
         detected = true;
      }
   }
}


__global__ void detectStrongClassifierGpu(
      cv::gpu::PtrStepSz<int32_t> integralImage,
      const uint32_t imageWidth,
      const uint32_t imageHeight,
      const uint8_t * const allClassifierData,
      const GpuStrongClassifier::Stage * const stages,
      const uint32_t stageCount,
      double * results)
{
   //const uint8_t * const allClassifierData = &g_FeatureData[0];
   assert(allClassifierData);
   assert(stages);
   assert(results);

   const uint32_t pixelCount = imageWidth * imageHeight;
   const uint32_t pixelIdx =  blockIdx.x * blockDim.x + threadIdx.x;

   if (!(pixelCount > pixelIdx))
   {
      return;
   }

   /*
   const uint32_t y = pixelIdx / integralImage.cols;
   const uint32_t x = pixelIdx - y * integralImage.cols;
   */

   const uint32_t y = pixelIdx / imageWidth;
   const uint32_t x = pixelIdx - y * imageWidth;

// FIXME remove this. Just for debugging
//if (x != 684 || y != 59 )
//   return;

   const int32_t * integralImageData = (int32_t *)(integralImage.data);
   bool detected = false;
   double hSum = 0.0;

   detectStrongClassifierAtPoint(
         integralImageData,
         imageWidth,
         imageHeight,
         integralImage.step / sizeof(uint32_t),
         x, y,
         allClassifierData,
         stages,
         stageCount,
         detected,
         hSum
         );

// FIXME remove this
//printf("Stage Threshold %f\n", stage.mStageThreshold);
//printf("Stage %d done x:%d y:%d\n\n\n",stageIdx,x,y);


   //printf("Match x:%d y:%d\n",x,y);
   results[pixelIdx] = hSum;
}

__global__ void detectStrongClassifierOnImageSetGpu(
      const int32_t * const integralImages,
      const uint32_t startImageIdx,
      const uint32_t imageCount,
      const uint32_t imageWidth,
      const uint32_t imageHeight,
      const uint8_t * const allClassifierData,
      const GpuStrongClassifier::Stage * const stages,
      const uint32_t stageCount,
      bool * allDetected)
{
   assert(integralImages);
   assert(allClassifierData);
   assert(stages);
   assert(allDetected);

   const uint32_t imageIdx =  blockIdx.x * blockDim.x + threadIdx.x;

   if (!(imageIdx < imageCount))
   {
      return;
   }

   const uint32_t pixelCountPerImage = imageHeight * imageWidth;
   const int32_t * const  integralImageData = integralImages + pixelCountPerImage * (imageIdx + startImageIdx);
   bool detected = false;
   double hSum = 0.0;

   detectStrongClassifierAtPoint(
         integralImageData,
         imageWidth,
         imageHeight,
         imageWidth,
         0, 0,
         allClassifierData,
         stages,
         stageCount,
         detected,
         hSum
         );

   allDetected[imageIdx] = detected;
}

bool Classifier::detectStrongClassifier(
      const std::vector<Classifier::Stage> & strongClassifier,
      FeatureTypes & featureTypes,
      const cv::gpu::GpuMat & gpuIntegralImage,
      std::vector<Classifier::ClassificationResult> & results
      )
{
   bool detected = false;
   double * resultsPtr = NULL;
   const uint32_t pixelCount = gpuIntegralImage.cols * gpuIntegralImage.rows;

   uint32_t strongClassifierXmin;
   uint32_t strongClassifierYmin;
   uint32_t strongClassifierXmax;
   uint32_t strongClassifierYmax;
   Classifier::sizeStrongClassifier(strongClassifier, featureTypes, strongClassifierXmin, strongClassifierYmin, strongClassifierXmax, strongClassifierYmax);

   CUDA_CHECK_RETURN(cudaMalloc(
         &resultsPtr,
         sizeof(double) * pixelCount)
         );

   const GpuStrongClassifier gpuStrongClassifier(strongClassifier);
   const uint32_t threadCount = 256;
   const uint32_t blockCount = (pixelCount + threadCount - 1) / threadCount;

   uint8_t * gpuFeatureData = FeatureTypes::getConstantFeatureData();

   cudaEvent_t start;
   cudaEvent_t stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   cudaEventRecord(start);

   detectStrongClassifierGpu<<<blockCount, threadCount>>>(
         gpuIntegralImage,
         gpuIntegralImage.cols,
         gpuIntegralImage.rows,
         gpuFeatureData,
   //      featureTypes.getGpuData(),
         gpuStrongClassifier.getGpuStages(),
         gpuStrongClassifier.mStagesCount,
         resultsPtr);

   CUDA_CHECK_RETURN(cudaPeekAtLastError());
   CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
   CUDA_CHECK_RETURN(cudaGetLastError());

   cudaEventRecord(stop);
   cudaEventSynchronize(stop);

   dumpElapsedTime("detectStrongClassifier:", start, stop);
   dumpFreeMemory("detectStrongClassifier:");


   // DEBUG
   double * hostResult = new double[pixelCount];
   CUDA_CHECK_RETURN(cudaMemcpy(hostResult, resultsPtr, sizeof(double) * pixelCount, cudaMemcpyDeviceToHost));

   for (uint32_t i = 0; i < pixelCount; ++i)
   {
      if (hostResult[i] > 0.0)
      {
         uint32_t y = i / gpuIntegralImage.cols;
         uint32_t x = i - gpuIntegralImage.cols * y;
         // std::cout << "detectStrongClassifier: at x:" << x << " y:" << y << std::endl;
         Classifier::ClassificationResult classificationResult;
         classificationResult.x = x; // x + strongClassifierXmin;
         classificationResult.y = y; // y + strongClassifierYmin;
         classificationResult.height = strongClassifierYmax - strongClassifierYmin;
         classificationResult.width = strongClassifierXmax - strongClassifierXmin;
         classificationResult.strength = hostResult[i];
         addUniqueResult(classificationResult, results);
      }
   }
   ////////////////

   CUDA_CHECK_RETURN(cudaFree(resultsPtr));
   return detected;
}

void Classifier::detectStrongClassifierOnImageSet(
      const std::vector<Classifier::Stage> & strongClassifier,
      FeatureTypes & featureTypes,
      const int32_t * const gpuIntegralImages,
      const uint32_t startImageIdx,
      const uint32_t imageCount,
      const uint32_t imageWidth,
      const uint32_t imageHeight,
      bool * results
      )
{
#ifdef DEBUG
         std::cout << "Debug: detectStrongClassifierOnImageSet startImageIdx:" << startImageIdx
               << ", imageCount:" << imageCount
               << ", imageWidth:" << imageWidth
               << ", imageHeight:" << imageHeight
               << ", stageCount:" << strongClassifier.size()
               << std::endl;
#endif

   bool * resultsGpu = NULL;

   CUDA_CHECK_RETURN(cudaMalloc(
         &resultsGpu,
         sizeof(bool) * imageCount)
         );

   const GpuStrongClassifier gpuStrongClassifier(strongClassifier);
   const uint32_t threadCount = 256;
   const uint32_t blockCount = (imageCount + threadCount - 1) / threadCount;
   uint8_t * gpuFeatureData = FeatureTypes::getConstantFeatureData();

   cudaEvent_t start;
   cudaEvent_t stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start);

   detectStrongClassifierOnImageSetGpu<<<blockCount, threadCount>>>(
         gpuIntegralImages,
         startImageIdx,
         imageCount,
         imageWidth,
         imageHeight,
         gpuFeatureData,
         gpuStrongClassifier.getGpuStages(),
         gpuStrongClassifier.mStagesCount,
         resultsGpu);

   CUDA_CHECK_RETURN(cudaPeekAtLastError());
   CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
   CUDA_CHECK_RETURN(cudaGetLastError());

   cudaEventRecord(stop);
   cudaEventSynchronize(stop);

   dumpElapsedTime("detectStrongClassifierOnImageSet:", start, stop);
   dumpFreeMemory("detectStrongClassifierOnImageSet:");

   CUDA_CHECK_RETURN(
         cudaMemcpy(
               results,
               resultsGpu,
               sizeof(bool) * imageCount,
               cudaMemcpyDeviceToHost));

   CUDA_CHECK_RETURN(cudaFree(resultsGpu));
}


void Classifier::addUniqueResult(const Classifier::ClassificationResult & newResult, std::vector<Classifier::ClassificationResult> & results)
{
   bool isUnique = true;

   // https://stackoverflow.com/questions/9324339/how-much-do-two-rectangles-overlap
   // SI = Max(0, Max(XA2, XB2) - Min(XA1, XB1)) * Max(0, Max(YA2, YB2) - Min(YA1, YB1))
   // SU = SA + SB - SI
   // ratio = SI / SU
   const int32_t xa1 = newResult.x;
   const int32_t xa2 = xa1 + newResult.width;

   const int32_t ya1 = newResult.y;
   const int32_t ya2 = ya1 + newResult.height;

   const int32_t areaA = newResult.width * newResult.height;

   for (std::vector<Classifier::ClassificationResult>::const_iterator resultIter = results.begin();
        resultIter != results.end();
        ++resultIter)
   {
      const int32_t xb1 = (*resultIter).x;
      const int32_t xb2 = xb1 + (*resultIter).width;

      const int32_t yb1 = (*resultIter).y;
      const int32_t yb2 = yb1 + (*resultIter).height;

      const int32_t areaB = (*resultIter).width * (*resultIter).height;

      const int32_t areaIntersect = max(0, max(xa2, xb2) - min(xa1, xb1)) * max(0, max(ya2, yb2) - min(ya1, yb1));
      const int32_t areaUnion = areaA + areaB - areaIntersect;

      double ratio = 1.0;

      if (areaUnion != 0.0)
      {
         ratio = areaIntersect / areaUnion;
      }

      if (ratio > 0.5)
      {
         isUnique = false;
         break;
      }
   }

   if (isUnique)
   {
      results.push_back(newResult);
   }
}
/*
texture<int32_t, 2> & Classifier::getTexIntegralImage()
{
   return texIntegralImage;
}
*/
