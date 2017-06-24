/*
 *
 *  Created on: May 20, 2017
 *      Author: Mario Lüder
 *
 */

#ifndef IMAGE_CUH_
#define IMAGE_CUH_

#include <string>
#include <vector>
#include <stdint.h>

#include <opencv2/core/gpumat.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "Classifier.cuh"

class Image
{
public:
/*
   struct CalcStatus
   {
      CalcStatus() : x(0), lastThreadIdx(0), y(0){}
      uint32_t x;
      uint32_t lastThreadIdx;
      uint32_t y;
   };
*/
   static bool fromFile(const std::string & fileName, Image & image);
   uint32_t getWidth() {return mImageWidth;}
   uint32_t getHeight() {return mImageHeight; }
   cv::gpu::GpuMat & getGpuIntegralImage() {return mGpuIntegralImage;}

   void displayClassificationResult(const std::vector<Classifier::ClassificationResult> & classificationResults);

   static void displayImageFalseColor(const cv::Mat & img);
   static void displayImage(const cv::Mat & img);
private:
   void calcIntegralImage(const uint32_t threadCount);
   void normalizeImage();

   std::string mFileName;

   uint32_t mImageWidth;
   uint32_t mImageHeight;

   cv::Mat mImage;
   cv::Mat mIntegralImage;
   cv::gpu::GpuMat mGpuImage;
   cv::gpu::GpuMat mGpuIntegralImage;
   cv::gpu::PtrStepSz<uchar>   mImagePtr;
   cv::gpu::PtrStepSz<int32_t> mIntegralImagePtr;
   cv::gpu::PtrStepSz<uchar>   * mGpuImagesPtr;
   cv::gpu::PtrStepSz<int32_t> * mGpuIntegralImagePtr;
};

#endif /* IMAGE_CUH_ */
