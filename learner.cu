/*
 *
 *  Created on: May 17, 2017
 *      Author: Mario Lüder
 *
 */

#include <stdio.h>
#include <stdlib.h>

#include "Learner.cuh"
#include "Classifier.cuh"
#include "FeatureTypes.cuh"
#include "FeatureValues.cuh"
#include "Image.cuh"
#include "IntegralImage.cuh"

// load image includes
#include <opencv2/core/core.hpp>
#include <opencv2/core/gpumat.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_profiler_api.h>

#include "utilities.cuh"
#include "defines.cuh"

void defineFeature(std::vector<FeatureType> & features)
{
   FeatureType featureTypeEdgeHorizontal(6, 2);
   featureTypeEdgeHorizontal.addRow() << 1;
   featureTypeEdgeHorizontal.addRow() << -1;

   FeatureType featureTypeEdgeVertical(2, 6);
   featureTypeEdgeVertical.addRow() << 1 << -1;

   FeatureType featureTypeLineHorizontal(6, 2);
   featureTypeLineHorizontal.addRow() << 1;
   featureTypeLineHorizontal.addRow() << -1;
   featureTypeLineHorizontal.addRow() << 1;

   FeatureType featureTypeLineVertical(2, 6);
   featureTypeLineVertical.addRow() << 1 << -1 << 1;

   features.push_back(featureTypeEdgeHorizontal);
   features.push_back(featureTypeEdgeVertical);
   features.push_back(featureTypeLineHorizontal);
   features.push_back(featureTypeLineVertical);

   /*
    # defines the feature
    # area white [width, height, color]
    # area black [width, height, color]
    featureTypeEdgeHorizontal = [
    [[8,2,1]],
    [[8,2,-1]]
    ]

    allClassifier += [[featureTypeEdgeHorizontal]]

    featureTypeEdgeVertical = [
    [[2,8,1],[2,8,-1]]
    ]

    allClassifier += [[featureTypeEdgeVertical]]

    featureTypeLineHorizontal = [
    [[6, 2, -1]],
    [[6, 2, 1]],
    [[6, 2, -1]]
    ]

    allClassifier += [[featureTypeLineHorizontal]]

    featureTypeLineVertical = [
    [[2, 6, -1],[2, 6, 1],[2, 6, -1]]
    ]
    */
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void)
{
   CUDA_CHECK_RETURN(cudaDeviceReset());

   // Set flag to enable zero copy access
   CUDA_CHECK_RETURN(cudaSetDeviceFlags(cudaDeviceMapHost));

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   const std::string outputFileName =
         "/mnt/project-disk/src/ObjectRecognition/selectedClassifier.txt";
   std::ofstream outputfile;
   outputfile.open(outputFileName.c_str());

   // settings
   const uint32_t ratioX = 1;
   const uint32_t ratioY = 1;
   const double classifierScale = 1.25;

   const uint32_t stages = 10;
   const double f = 0.3;  // false positive rate per layer
   const double d = 0.99; // minimum acceptable detection rate per layer
   const double fTarget = pow(f, stages); // overall false positive rate

   // faces
   // const std::string pathPositiveImages = "/mnt/project-disk/src/ObjectRecognition/data/facesTraining/att_faces/*.pgm";
   // const std::string pathNegativeImages = "/mnt/project-disk/src/ObjectRecognition/data/facesTraining/negatives-small-norm/*.png";

   // faces small subset
   //const std::string pathPositiveImages = "/mnt/project-disk/src/ObjectRecognition/data/facesTraining/att_faces_subset/*.pgm";
   //const std::string pathNegativeImages = "/mnt/project-disk/src/ObjectRecognition/data/facesTraining/negatives-small_subset/*.png";

   // cars
   const std::string pathPositiveImages =
         "/mnt/project-disk/src/ObjectRecognition/data/cars/TheKITTIVision/training/image_back_inline/*.png";
   const std::string pathNegativeImages =
         "/mnt/project-disk/src/ObjectRecognition/data/cars/negatives-64x64-norm/*.png";

   size_t mem_tot_0 = 0;
   size_t mem_free_0 = 0;
   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << "Free memory:" << mem_free_0 << " Mem total: " << mem_tot_0
         << std::endl;

   cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 600);

   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << "Free memory:" << mem_free_0 << " Mem total: " << mem_tot_0
         << std::endl;

   // load images
   std::vector<cv::Mat> images;
   std::vector<cv::gpu::GpuMat> gpuImages;
   std::vector<std::string> fileNamesPos;
   std::vector<std::string> fileNamesNeg;
   std::vector<std::string> fileNames;
   std::vector<cv::gpu::PtrStepSz<uchar> > imagePtrsVector;

   cv::gpu::PtrStepSz<uchar> * gpuImagesPtr;

   // load positive and negative images
   cv::glob(pathPositiveImages, fileNamesPos, true);
   cv::glob(pathNegativeImages, fileNamesNeg, true);

   assert(fileNamesPos.size() > 0);
   assert(fileNamesNeg.size() > 0);

   fileNames.insert(fileNames.end(), fileNamesPos.begin(), fileNamesPos.end());
   fileNames.insert(fileNames.end(), fileNamesNeg.begin(), fileNamesNeg.end());

   cv::Mat firstImage = cv::imread(*fileNames.begin(), CV_LOAD_IMAGE_GRAYSCALE);
   const uint32_t imageWidth = firstImage.cols;
   const uint32_t imageHeight = firstImage.rows;
   uint32_t countImages = fileNames.size();
   assert(countImages > 0);

   uchar * imagesMem = NULL;
   imagesMem =
         new uchar[countImages * sizeof(uchar) * imageHeight * imageWidth];

   uchar * imagesGpuMem = NULL;
   CUDA_CHECK_RETURN(
         cudaMalloc((void** )&imagesGpuMem,
               countImages * sizeof(uchar) * imageHeight * imageWidth));

   int32_t * integralImagesGpuMem = NULL;

   CUDA_CHECK_RETURN(
         cudaMalloc((void** )&integralImagesGpuMem,
               countImages * sizeof(int32_t) * imageHeight * imageWidth));

   // pointer to integral images (index)
   std::vector<uint32_t> availableIntegralImages;
   availableIntegralImages.reserve(countImages);

#ifdef DEBUG
   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << "Debug: Reserved space for images" << std::endl;
   std::cout << "Debug: Free: " << mem_free_0 << " Mem total: " << mem_tot_0
         << std::endl;
#endif
   bool fileError = false;
   for (uint32_t i = 0; i < countImages; ++i)
   {
      const std::string & fileName = fileNames[i];
      uchar * imagePtr = &imagesMem[imageHeight * imageWidth * i];
      uchar * imageGpuPtr = &imagesGpuMem[imageHeight * imageWidth * i];

      cv::Mat image(imageHeight, imageWidth, CV_8U, imagePtr);
      cv::gpu::GpuMat gpuImage(imageHeight, imageWidth, CV_8U, imageGpuPtr);

      int32_t * integralImageGpuPtr = &integralImagesGpuMem[imageHeight
            * imageWidth * i];

      cv::gpu::GpuMat gpuIntegralImage(imageHeight, imageWidth, CV_32S,
            integralImageGpuPtr);

      cv::Mat imageReadMat = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

      if (imageReadMat.empty())
      {
         std::cout << "Error: File:" << fileName << " is broken" << std::endl;
         fileError = true;
      }

      imageReadMat.copyTo(image);
      // cvtColor(image,image,CV_BGR2GRAY);

      images.push_back(image);

      gpuImage.upload(image);
      gpuImages.push_back(gpuImage);

      const cv::gpu::PtrStepSz<uchar> imageMatPtr = gpuImages.back();

      imagePtrsVector.push_back(imageMatPtr);
      availableIntegralImages.push_back(i);
   }

   if (fileError)
   {
      return 1;
   }

#ifdef DEBUG
   std::cout << "Debug: Count positive images:" << fileNamesPos.size()
         << std::endl;
   const uint32_t countPosImages = fileNamesPos.size();
   std::cout << "Debug: Count negative images:" << fileNamesNeg.size()
         << std::endl;

   std::cout << "Debug: Ratio X:" << ratioX << " Y:" << ratioY << std::endl;
#endif

   CUDA_CHECK_RETURN(
         cudaMalloc((void** )&gpuImagesPtr,
               countImages * sizeof(cv::gpu::PtrStepSz<uchar>)));

   CUDA_CHECK_RETURN(
         cudaMemcpy(gpuImagesPtr, &imagePtrsVector[0],
               countImages * sizeof(cv::gpu::PtrStepSz<uchar>),
               cudaMemcpyHostToDevice));

#ifdef DEBUG
   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << "Debug: Calc integral images:" << countImages << std::endl;
   std::cout << "Debug: Free: " << mem_free_0 << " Mem total: " << mem_tot_0
         << std::endl;
#endif

   getIntegralImage<<<(countImages + WORK_SIZE - 1) / WORK_SIZE, WORK_SIZE>>>(
         countImages, gpuImagesPtr, integralImagesGpuMem);

   CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
   CUDA_CHECK_RETURN(cudaGetLastError());

#ifdef DEBUG
   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << "Debug: Calc integral images download Done!:" << std::endl;
   std::cout << "Debug: Free: " << mem_free_0 << " Mem total: " << mem_tot_0
         << std::endl;
#endif

   // we don't need the original images anymore - keep the integral images in memory
   std::vector<cv::Mat>().swap(images);
   delete[] imagesMem;
   CUDA_CHECK_RETURN(cudaFree(gpuImagesPtr));
   CUDA_CHECK_RETURN(cudaFree(imagesGpuMem));

   // copy the integral images indexes to gpu
   uint32_t * availableIntegralImagesGpu = NULL;

   CUDA_CHECK_RETURN(
         cudaMalloc((void** )&availableIntegralImagesGpu,
               countImages * sizeof(uint32_t)));

   CUDA_CHECK_RETURN(
         cudaMemcpy(availableIntegralImagesGpu, &availableIntegralImages[0],
               countImages * sizeof(uint32_t), cudaMemcpyHostToDevice));

#ifdef DEBUG
   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << "Debug: Cleared original images" << std::endl;
   std::cout << "Debug: Free memory: " << mem_free_0 << " Mem total: "
         << mem_tot_0 << std::endl;
#endif

   /*
    cv::Mat displayImage;
    gpuIntegralImages[1].download(displayImage);
    Image::displayImageFalseColor(displayImage);
    return 0;
    */

   // init weight for all images
   double * imageWeightsPtr = new double[countImages];
   double * gpuImageWeightsPtr = NULL;

   CUDA_CHECK_RETURN(
         cudaMalloc((void ** )&gpuImageWeightsPtr,
               countImages * sizeof(cv::gpu::PtrStepSz<uchar>)));

   double initWeight = 1.0 / countImages;

   for (uint32_t i = 0; i < countImages; ++i)
   {
      imageWeightsPtr[i] = initWeight;
   }

   cudaMemcpy(gpuImageWeightsPtr, imageWeightsPtr, countImages * sizeof(double),
         cudaMemcpyHostToDevice);

   const uint32_t pixelCount = (imageWidth / ratioX) * (imageHeight / ratioY);
   //const uint32_t pixelCount = 5000;

#ifdef DEBUG
   std::cout << "Debug: Image width x height:(" << imageWidth << "x"
         << imageHeight << ")" << std::endl;
#endif

   // generate all classifier in all scales
   FeatureTypes featureTypes;
   defineFeature(featureTypes);
   featureTypes.generateClassifier(classifierScale, imageWidth, imageHeight,
         true);

   // reserve space for the result
   Classifier::SelectionResult * results = NULL;
   CUDA_CHECK_RETURN(
         cudaHostAlloc((void ** )&results,
               pixelCount * sizeof(Classifier::SelectionResult),
               cudaHostAllocMapped));
   assert(results);

   Classifier::SelectionResult * gpuResults = NULL;
   CUDA_CHECK_RETURN(
         cudaHostGetDevicePointer((void ** )&gpuResults, (void * )results, 0));

   // cascade learner
   double Fcurr = 1.0;
   double Fprev = 1.0;
   double Dcurr = 1.0;
   double Dprev = 1.0;

   uint32_t stageIdx = 0;
   std::vector<Classifier::Stage> classifierStages;

#ifdef DEBUG
   std::cout << "Debug: detection rate d: " << d;
   std::cout << " False positive Rate f: " << f;
   std::cout << " Targeted false positive rate: " << fTarget;
   std::cout << std::endl;
#endif

   while (Fcurr > fTarget)
   {
      Classifier::Stage classifierStage;
      std::vector<GetFeatureValueResult *> featureValuesPtrs;

      stageIdx++;

#ifdef DEBUG
      std::cout << "Debug: Stage:" << stageIdx << std::endl;

#endif

      uint32_t roundIdx = 0;
      Fcurr = Fprev;
      Dcurr = Dprev;
      std::vector<uint32_t> falsePositiveIdx;

#ifdef DEBUG
      std::cout << "Debug: (f * Fprev):" << (f * Fprev);
      std::cout << " Fi: " << Fcurr;
      std::cout << " Di: " << Dcurr;
      std::cout << std::endl;
#endif

      while (Fcurr > (f * Fprev))
      {
         roundIdx++;
         uint32_t newClassifierCount = 0;

         adaBoost(roundIdx, countImages, countPosImages, pixelCount, ratioX,
               ratioY, imageWidth, imageHeight, featureTypes,
               integralImagesGpuMem, availableIntegralImagesGpu,
               imageWeightsPtr, gpuImageWeightsPtr, start, stop,
               classifierStage.stagedClassifier, results, gpuResults,
               newClassifierCount, outputfile);

         for (uint32_t i = classifierStage.stagedClassifier.size()
               - newClassifierCount;
               i < classifierStage.stagedClassifier.size(); ++i)
         {
            if (classifierStage.stagedClassifier[i].error != 0.0)
            {
               updateImageWeights(countImages, countPosImages,
                     integralImagesGpuMem, availableIntegralImagesGpu, imageWidth,
                     imageHeight, featureTypes,
                     classifierStage.stagedClassifier[i], imageWeightsPtr,
                     classifierStage.betas, featureValuesPtrs);

               cudaMemcpy(gpuImageWeightsPtr, imageWeightsPtr,
                     countImages * sizeof(double), cudaMemcpyHostToDevice);
            }
         }

         if (newClassifierCount == 0)
         {
#ifdef DEBUG
            std::cout << "Debug: No new Classifier found" << std::endl;
#endif
            break;
         }

         evaluateClassifier(countImages, countPosImages,
               classifierStage.stagedClassifier, featureValuesPtrs,
               classifierStage.betas, d, Dprev, Fcurr, Dcurr, falsePositiveIdx,
               classifierStage.stageThreshold);
      }

      if (classifierStage.stagedClassifier.empty())
      {
#ifdef DEBUG
         std::cout << "Debug: Classifier stage is empty" << std::endl;
#endif
         break;
      }

      classifierStages.push_back(classifierStage);

      // free feature values
      for (std::vector<GetFeatureValueResult *>::const_iterator featureValuesPtrIter =
            featureValuesPtrs.begin();
            featureValuesPtrIter != featureValuesPtrs.end();
            ++featureValuesPtrIter)
      {
         CUDA_CHECK_RETURN(cudaFreeHost(*featureValuesPtrIter));
      }

      featureValuesPtrs.clear();

      if (falsePositiveIdx.size() == 0)
      {
#ifdef DEBUG
         std::cout << "Debug: No false positive images are left. Finishing!"
               << std::endl;
#endif
         break;
      }

      // create a new set of positive images and false positive images
      std::vector<uint32_t> availableIntegralImagesOld;
      std::swap(availableIntegralImages, availableIntegralImagesOld);
      availableIntegralImages.clear();
      availableIntegralImages.reserve(countPosImages + falsePositiveIdx.size());

      for (uint32_t i = 0; i < countPosImages; ++i)
      {
         availableIntegralImages.push_back(i);
      }

      for (uint32_t falsePositiveIdxIter = 0;
            falsePositiveIdxIter < falsePositiveIdx.size();
            ++falsePositiveIdxIter)
      {
         availableIntegralImages.push_back(
               availableIntegralImagesOld[
                                          falsePositiveIdx[falsePositiveIdxIter]
                                          ]
               );
      }

      countImages = availableIntegralImages.size();

      CUDA_CHECK_RETURN(
            cudaMemcpy(availableIntegralImagesGpu, &availableIntegralImages[0],
                  countImages * sizeof(int32_t), cudaMemcpyHostToDevice));

#ifdef DEBUG
      std::cout << "Debug: Images left:" << countImages << std::endl;
#endif

      // init image weights
      double initWeight = 1.0 / countImages;

      for (uint32_t i = 0; i < countImages; ++i)
      {
         imageWeightsPtr[i] = initWeight;
      }

      Fprev = Fcurr;
      Dprev = Dcurr;
   }

   std::cout << " --- END --- " << std::endl;
   std::stringstream prettyClassifier;
   prettyClassifier << "[";

   for (uint32_t classifierStageIdx = 0;
         classifierStageIdx < classifierStages.size(); ++classifierStageIdx)
   {
      const Classifier::Stage & classifierStage =
            classifierStages[classifierStageIdx];

      prettyClassifier << "[[";

      for (uint32_t i = 0; i < classifierStage.stagedClassifier.size(); ++i)
      {
         prettyClassifier
               << Classifier::dumpSelectedClassifier(
                     classifierStage.stagedClassifier[i], featureTypes);

         if ((i + 1) < classifierStage.stagedClassifier.size())
         {
            prettyClassifier << ",";
         }
      }

      prettyClassifier << "]," << std::endl;

      // print betas
      prettyClassifier << "[";
      for (uint32_t i = 0; i < classifierStage.betas.size(); ++i)
      {
         prettyClassifier << classifierStage.betas[i];
         if ((i + 1) < classifierStage.betas.size())
         {
            prettyClassifier << ",";
         }
      }
      prettyClassifier << "]," << std::endl;
      prettyClassifier << classifierStage.stageThreshold << "]";

      if ((classifierStageIdx + 1) < classifierStages.size())
      {
         prettyClassifier << ",";
      }

      prettyClassifier << std::endl;
   }

   prettyClassifier << "]" << std::endl;
   std::cout << prettyClassifier.str();

   outputfile << prettyClassifier;
   outputfile.close();

   CUDA_CHECK_RETURN(cudaFree(integralImagesGpuMem));
   CUDA_CHECK_RETURN(cudaFree(availableIntegralImagesGpu));
   delete[] imageWeightsPtr;
   CUDA_CHECK_RETURN(cudaFree(gpuImageWeightsPtr));
   //CUDA_CHECK_RETURN(cudaDeviceReset());
   return 0;
}
