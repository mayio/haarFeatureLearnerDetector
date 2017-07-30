/*
 *
 *  Created on: May 17, 2017
 *      Author: Mario Lüder
 *
 */

#ifndef INTEGRAL_IMAGE_CUH_
#define INTEGRAL_IMAGE_CUH_

/*
__global__ void getIntegralImage(uint32_t countImages,
      cv::cuda::PtrStepSz<uchar> originalImages[],
      cv::cuda::PtrStepSz<int32_t> integralImages[])
{
   const uint32_t imageIdx = blockIdx.x * blockDim.x + threadIdx.x;

   if (countImages <= imageIdx)
      return;

   cv::cuda::PtrStepSz<uchar> & originalImage = originalImages[imageIdx];
   cv::cuda::PtrStepSz<int32_t> & integralImage = integralImages[imageIdx];

   for (uint32_t y = 0; y < originalImage.rows; ++y)
   {
      for (uint32_t x = 0; x < originalImage.cols; ++x)
      {
         int32_t i1 = 0;
         int32_t i2 = 0;
         int32_t i4 = 0;

         if (y > 0)
            i1 = integralImage(y - 1, x);

         if (x > 0)
            i2 = integralImage(y, x - 1);

         int i3 = originalImage(y, x);

         if ((x > 0) && (y > 0))
            i4 = integralImage(y - 1, x - 1);

         integralImage(y, x) = i1 + i2 + i3 - i4;

      }
   }
}
*/

__global__ void getIntegralImage(
      uint32_t countImages,
      cv::cuda::PtrStepSz<uchar> originalImages[],
      int32_t * integralImages
      )
{
   const uint32_t imageIdx = blockIdx.x * blockDim.x + threadIdx.x;

   if (countImages <= imageIdx)
      return;

   cv::cuda::PtrStepSz<uchar> & originalImage = originalImages[imageIdx];

   uint8_t * originalImageData = (uint8_t *)(originalImage.data);
   int32_t * integralImageData = integralImages + imageIdx * originalImage.rows * originalImage.cols;

   int32_t i1 = 0;
   int32_t i2 = 0;
   int32_t i4 = 0;

   int32_t *       integralImagePtrY1  = NULL;
   for (uint32_t y = 0; y < originalImage.rows; ++y)
   {
      const uint8_t * const originalImagePtrY  = originalImageData + y * originalImage.step;
      int32_t *             integralImagePtrY  = integralImageData + y * originalImage.cols;

      for (uint32_t x = 0; x < originalImage.cols; ++x)
      {
         if (y > 0)
            i1 = integralImagePtrY1[x];

         int i3 = originalImagePtrY[x];

         i2 = i1 + i2 + i3 - i4;
         integralImagePtrY[x] = i2;

         i4 = i1;

/*
         i4 i1
         i2 i3
*/

      }

      i4 = 0;
      i2 = 0;
      integralImagePtrY1 = integralImagePtrY;
   }
}

#endif
