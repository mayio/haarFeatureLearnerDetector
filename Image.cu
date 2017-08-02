/*
 *
 *  Created on: May 20, 2017
 *      Author: Mario Lüder
 *
 */

#include "Image.cuh"
#include "utilities.cuh"

#include "stdio.h"
#include <iostream>

// load image includes
// #include <opencv2/contrib/contrib.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/core/gpumat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <cuda_profiler_api.h>

#include <sys/time.h>
#include <unistd.h>

// texture definition
texture<uint8_t, 2> texOriginalImage;


__global__ void getSingleIntegralImage(
//      cv::gpu::PtrStepSz<uchar> originalImage,
      cv::cuda::PtrStepSz<int32_t> integralImage,
      const uint32_t imageWidth,
      const uint32_t imageHeight,
      volatile uint32_t * blockSync
      )
{
   const uint32_t y =  blockIdx.x * blockDim.x + threadIdx.x;

   if (y >= imageHeight)
   {
      return;
   }

   extern __shared__ uint32_t progress[];

   if (threadIdx.x == 0 && blockIdx.x == 0)
   {
      progress[threadIdx.x] = imageWidth;
   }
   else
   {
      progress[threadIdx.x] = 0;
   }

   bool isLastThreadInBlock = false;
   bool isVeryLastThread    = false;

   if (blockDim.x - 1 == threadIdx.x)
   {
      isLastThreadInBlock = true;
      blockSync[blockIdx.x] = 0;
      isVeryLastThread = (imageHeight - 1 == y) ? true : false;
   }

   __syncthreads();

   // uint8_t ** originalImageData = (uint8_t **)(originalImage.data);
   int32_t ** integralImageData = (int32_t **)(integralImage.data);

   // const uint8_t * const originalImagePtrY  =            ((uint8_t *)(originalImageData) + y * originalImage.step);
   int32_t *             integralImagePtrY  = (int32_t *)((uint8_t *)(integralImageData) + y * integralImage.step);
   const int32_t *       integralImagePtrY1 = (int32_t *)((uint8_t *)(integralImagePtrY) - integralImage.step);

   int32_t i2 = 0;
   int32_t i4 = 0;

   for (uint32_t x = 0; x < imageWidth; )
   {
      //printf("Thread:%d start x:%d progress:%d\n ", threadIdx.x, x, progress[threadIdx.x]);

      if ((threadIdx.x == 0) && (blockIdx.x > 0) && (progress[threadIdx.x] <= x))
      {
         uint32_t pVal = 0;
         do
         {
            pVal = blockSync[blockIdx.x - 1];
         }
         while (pVal <= x);

         //printf("Block:%d start x:%d\n", blockIdx.x, x);
         progress[threadIdx.x] = pVal;
      }

      __syncthreads();

      if (progress[threadIdx.x] > x)
      {
         //printf("x:%d y:%d\n", x, y);

         int32_t i1 = 0;

         if (y > 0)
            i1 = integralImagePtrY1[x];

         //int i3 = originalImagePtrY[x];
         int i3 = tex2D(texOriginalImage, x, y); // texture access

         integralImagePtrY[x] = i1 + i2 + i3 - i4;

         i4 = i1;
         i2 = integralImagePtrY[x];

         x++;

         if (isLastThreadInBlock)
         {
            if (!isVeryLastThread)
            {
               blockSync[blockIdx.x] = x;
               //printf("Block:%d done x:%d\n", blockIdx.x, x);
            }
         }
         else
         {
            progress[threadIdx.x + 1] = x;
         }
      }
   }
}
__global__ void getHistogram(
      cv::cuda::PtrStepSz<uchar> originalImage,
      const uint32_t imageWidth,
      const uint32_t imageHeight,
      uint32_t * histogram
      )
{
   __shared__ uint32_t tempHistogram[256];
   tempHistogram[threadIdx.x] = 0;

   __syncthreads();

   const uint32_t x =  blockIdx.x * blockDim.x + threadIdx.x;

   uint8_t * originalImageData = (uint8_t *)(originalImage.data);
   uint32_t y = 0;

   if (x < imageWidth)
   {
      while(y < imageHeight)
      {
         atomicAdd(&tempHistogram[originalImageData[x]], 1);
         originalImageData += originalImage.step;
         y++;
      }
   }

   __syncthreads();

   atomicAdd(&(histogram[threadIdx.x]), tempHistogram[threadIdx.x]);
}

__global__ void normalizeImageGpu(
      cv::cuda::PtrStepSz<uchar> originalImage,
      const uint32_t imageWidth,
      const uint32_t imageHeight,
      uint32_t * lookUpTable
      )
{
   const uint32_t i =  blockIdx.x * blockDim.x + threadIdx.x;

   const uint32_t y = i / imageWidth;
   const uint32_t x = i - y * imageWidth;

   uint8_t * originalImageData =  (uint8_t *)(originalImage.data);
   uint8_t * originalImagePtrY = ((uint8_t *)(originalImageData) + y * originalImage.step);

   originalImagePtrY[x] = lookUpTable[originalImagePtrY[x]];
}

bool Image::fromFile(const std::string & fileName, Image & image)
{
   cudaEvent_t start;
   cudaEvent_t stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   bool success = false;
/*
   for (uint32_t threadCount = 2; threadCount <= 1024; threadCount *= 2)
   {

*/
   uint32_t threadCount = 128;
/**/
         image.mImage = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
         //cudaEventRecord(start);
         // cv::normalize(image.mImage, image.mImage, 0, 255, cv::NORM_MINMAX);
         //cudaEventRecord(stop);
         //cudaEventSynchronize(stop);

         //dumpElapsedTime("fromFile: normalize image", start, stop);
         //image.mImage.convertTo(image.mImage, CV_8U);

         image.mGpuImage.upload(image.mImage);

         image.mImageWidth = image.mImage.cols;
         image.mImageHeight = image.mImage.rows;
          //image.mImageWidth = 20;
          //image.mImageHeight = 10;
         //threadCount = image.mImageHeight;

         std::cout << "fromFile: ImageDimension: width:" << image.mImageWidth << " height:" << image.mImageHeight << std::endl;

         if (image.mImageWidth > 0 && image.mImageHeight > 0)
         {
            image.mGpuIntegralImage.create(image.mImageHeight, image.mImageWidth, CV_32S);
            image.normalizeImage();
            image.calcIntegralImage(threadCount);

/*
            Image::displayImage(image.mImage);
            image.normalizeImage();

            cv::Mat normalizedImage(image.mImageHeight, image.mImageWidth, CV_8U);
            image.mGpuImage.download(normalizedImage);
            Image::displayImage(normalizedImage);

            success = true;
            // CUDA_CHECK_RETURN(cudaMallocHost(&integralImagePtr, sizeof(int32_t) * image.mImageHeight * image.mImageWidth));

            // cv::Mat integralImage(image.mImageHeight, image.mImageWidth, CV_32S, integralImagePtr);
            cv::Mat integralImage(image.mImageHeight, image.mImageWidth, CV_32S);
            // cv::cuda::GpuMat gpuIntegralImage(image.mImageHeight, image.mImageWidth, CV_32S, integralImagePtr);
            //integralImage.create(image.mImageHeight, image.mImageWidth, CV_32S, cv::Scalar::all(255));
            //integralImage.zeros(image.mImageHeight, image.mImageWidth, CV_32S);
            //integralImage.setTo(cv::Scalar::all(255));

            image.mGpuIntegralImage.upload(integralImage);
            //image.mGpuIntegralImage = gpuIntegralImage;
            image.calcIntegralImage(threadCount);
*/
/* Show integral image
            cv::Mat integralImage;
            image.mGpuIntegralImage.download(integralImage);
            Image::displayImageFalseColor(integralImage);
*/
         }
/*
   }
*/
   return success;
}


void Image::calcIntegralImage(const uint32_t threadCount)
{
   cudaEvent_t start;
   cudaEvent_t stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   const uint32_t blockCountHeight = (mImageHeight + threadCount - 1) / threadCount;
   uint32_t * sharedMemProgress;
   uint32_t * blockSync;
   CUDA_CHECK_RETURN(cudaMalloc(&sharedMemProgress, sizeof(uint32_t) * (threadCount)));
   CUDA_CHECK_RETURN(cudaMalloc(&blockSync, sizeof(uint32_t) * (blockCountHeight)));

   // bind original image to texture
   cudaChannelFormatDesc desc = cudaCreateChannelDesc<uint8_t>();
   CUDA_CHECK_RETURN(cudaBindTexture2D(
         NULL,
         texOriginalImage,
         mGpuImage.data,
         desc,
         mGpuImage.cols,
         mGpuImage.rows,
         mGpuImage.step));

   std::cout << "calcIntegralImage: blockCountHeight:" << blockCountHeight << " threadCount:" << threadCount << " mImageWidth:" << mImageWidth << " mImageHeight:" << mImageHeight << std::endl;
   cudaEventRecord(start);
   getSingleIntegralImage<<<blockCountHeight, threadCount, sizeof(uint32_t) * threadCount>>>(/*mGpuImage, */mGpuIntegralImage, mImageWidth, mImageHeight, blockSync);

   CUDA_CHECK_RETURN(cudaPeekAtLastError());
   CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
   CUDA_CHECK_RETURN(cudaGetLastError());

   cudaEventRecord(stop);
   cudaEventSynchronize(stop);

   dumpElapsedTime("calcIntegralImage", start, stop);

   CUDA_CHECK_RETURN(cudaUnbindTexture(texOriginalImage));
   dumpFreeMemory("calcIntegralImage finished:");

   CUDA_CHECK_RETURN(cudaFree(sharedMemProgress));
   CUDA_CHECK_RETURN(cudaFree(blockSync));
}
void Image::displayImageFalseColor(const cv::Mat & img)
{
   double min;
   double max;
   cv::minMaxIdx(img, &min, &max);
   cv::Mat adjMap;
   // expand your range to 0..255. Similar to histEq();
   img.convertTo(adjMap,CV_8UC1, 255.0 / (max-min), -255.0 * min / (max-min));

   cv::Mat falseColorsMap;
   cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);

   cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
   cv::imshow("Display window", falseColorsMap);
   cvWaitKey();
}

void Image::displayImage(const cv::Mat & img)
{
   cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
   cv::imshow("Display window", img);
   cvWaitKey();
}

void Image::normalizeImage()
{
   cudaEvent_t start;
   cudaEvent_t stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   const uint32_t histogramSize = 256;
   const double rangeByPixelCount = 255.0 / (mImageHeight * mImageWidth);

   const uint32_t threadCount = histogramSize;
   const uint32_t blockCountWidth = (mImageWidth + threadCount - 1) / threadCount;
   uint32_t * gpuHistogram;
   uint32_t * histogram = new uint32_t[histogramSize];
   CUDA_CHECK_RETURN(cudaMalloc(&gpuHistogram, sizeof(uint32_t) * histogramSize));
   memset(histogram, 0, sizeof(uint32_t) * histogramSize);
   CUDA_CHECK_RETURN(cudaMemcpy(gpuHistogram, histogram, sizeof(uint32_t) * histogramSize, cudaMemcpyHostToDevice));

   std::cout << "normalizeImage: blockCountWidth:" << blockCountWidth << " threadCount:" << threadCount << " mImageWidth:" << mImageWidth << " mImageHeight:" << mImageHeight << std::endl;
   cudaEventRecord(start);
   getHistogram<<<blockCountWidth, threadCount, sizeof(uint32_t) * threadCount>>>(mGpuImage, mImageWidth, mImageHeight, gpuHistogram);

   CUDA_CHECK_RETURN(cudaPeekAtLastError());
   CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
   CUDA_CHECK_RETURN(cudaGetLastError());

   cudaEventRecord(stop);
   cudaEventSynchronize(stop);

   dumpElapsedTime("normalizeImage create histogram", start, stop);

   // create a look up table (LUT)
   cudaMemcpy(histogram, gpuHistogram, sizeof(uint32_t) * histogramSize, cudaMemcpyDeviceToHost);
   uint32_t histogramSum = 0;

   for (uint32_t i = 0; i < histogramSize; ++i)
   {
      histogramSum += histogram[i];
      histogram[i] = floor(rangeByPixelCount * histogramSum + 0.00001);
   }

   cudaMemcpy(gpuHistogram, histogram, sizeof(uint32_t) * histogramSize, cudaMemcpyHostToDevice);

   // apply LUT to image
   const uint32_t blockCount = (mImageWidth * mImageHeight + threadCount - 1) / threadCount;
   cudaEventRecord(start);
   normalizeImageGpu<<<blockCount, threadCount, sizeof(uint32_t) * threadCount>>>(mGpuImage, mImageWidth, mImageHeight, gpuHistogram);

   CUDA_CHECK_RETURN(cudaPeekAtLastError());
   CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
   CUDA_CHECK_RETURN(cudaGetLastError());

   cudaEventRecord(stop);
   cudaEventSynchronize(stop);

   dumpElapsedTime("normalizeImage apply LUT", start, stop);
   dumpFreeMemory("normalizeImage finished:");

   CUDA_CHECK_RETURN(cudaFree(gpuHistogram));
   delete[] histogram;
}

void Image::displayClassificationResult(const std::vector<Classifier::ClassificationResult> & classificationResults)
{
   // text settings
   int fontFace = cv::FONT_HERSHEY_SIMPLEX;
   double fontScale = 0.4;
   int thickness = 1;

   // image

   cv::Mat normalizedImage;
   mGpuImage.download(normalizedImage);

   cv::Mat resultImage;// = mImage.clone();
   cv::cvtColor(normalizedImage, resultImage, CV_GRAY2RGB);

   for (std::vector<Classifier::ClassificationResult>::const_iterator resultIter = classificationResults.begin();
        resultIter != classificationResults.end();
        ++resultIter)
   {
      cv::Point topLeft;
      topLeft.x = (*resultIter).x;
      topLeft.y = (*resultIter).y;
      cv::Point bottomRight;
      bottomRight.x = (*resultIter).x + (*resultIter).width;
      bottomRight.y = (*resultIter).y + (*resultIter).height;
      cv::Scalar color(0,255, 255);

      cv::rectangle(resultImage, topLeft, bottomRight, color);

      // then put the text itself
      std::stringstream textStream;
      textStream << "x:" << (*resultIter).x << " y:" << (*resultIter).y;
/*
      int baseline=0;
      cv::Size textSize = cv::getTextSize(textStream.str(), fontFace,
                                  fontScale, thickness, &baseline);
*/
      cv::putText(resultImage, textStream.str(), topLeft, fontFace, fontScale,
            color, thickness, 8);
   }


   Image::displayImage(resultImage);
}
