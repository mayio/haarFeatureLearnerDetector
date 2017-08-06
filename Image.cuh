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

//#include <opencv2/core/gpumat.hpp>
//#include <opencv2/gpu/gpu.hpp>

#include <opencv2/cudev/ptr2d/gpumat.hpp>

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
   ~Image()
   {
      release();
   }

   void release()
   {
      mImage.release();
      mIntegralImage.release();

      mGpuImage.release();
      mGpuIntegralImage.release();

      assert(!mImage.data);
      assert(!mIntegralImage.data);
      assert(!mGpuImage.data);
      assert(!mGpuIntegralImage.data);
   }

   static bool fromFile(const std::string & fileName, Image & image);
   uint32_t getWidth() {return mImageWidth;}
   uint32_t getHeight() {return mImageHeight; }
   cv::cuda::GpuMat & getGpuIntegralImage() {return mGpuIntegralImage;}

   void displayClassificationResult(const std::vector<Classifier::ClassificationResult> & classificationResults);

   void storeClassificationResult(
         const std::string & testImageBaseName,
         const std::vector<Classifier::ClassificationResult> & classificationResults,
         const std::string & classificationResultFolder,
         const uint32_t resultImageWidth,
         const uint32_t resultImageHeight);

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
   cv::cuda::GpuMat mGpuImage;
   cv::cuda::GpuMat mGpuIntegralImage;
   cv::cuda::PtrStepSz<uchar>   mImagePtr;
   cv::cuda::PtrStepSz<int32_t> mIntegralImagePtr;
};

#endif /* IMAGE_CUH_ */
