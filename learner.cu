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
   FeatureType featureTypeEdgeHorizontal(4, 2);
   featureTypeEdgeHorizontal.addRow() << 1;
   featureTypeEdgeHorizontal.addRow() << -1;

   FeatureType featureTypeEdgeVertical(2, 4);
   featureTypeEdgeVertical.addRow() << 1 << -1;

   FeatureType featureTypeLineHorizontal(4, 2);
   featureTypeLineHorizontal.addRow() << 1;
   featureTypeLineHorizontal.addRow() << -1;
   featureTypeLineHorizontal.addRow() << 1;

   FeatureType featureTypeLineVertical(2, 4);
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

bool loadImage(
      const std::string & fileName,
      const uint32_t loadImageIdx,
      const uint32_t imageWidth,
      const uint32_t imageHeight,
      uint8_t * imagesMem,
      uint8_t * imagesGpuMem,
      int32_t * integralImagesGpuMem,
      std::vector<cv::Mat> & images,
      std::vector<cv::gpu::GpuMat> & gpuImages,
      std::vector<cv::gpu::PtrStepSz<uchar> > & imagePtrsVector
)
{
   bool fileError = false;

   uchar * imagePtr    = &imagesMem[   imageHeight * imageWidth * loadImageIdx];
   uchar * imageGpuPtr = &imagesGpuMem[imageHeight * imageWidth * loadImageIdx];

   cv::Mat image(imageHeight, imageWidth, CV_8U, imagePtr);
   cv::gpu::GpuMat gpuImage(imageHeight, imageWidth, CV_8U, imageGpuPtr);

   int32_t * integralImageGpuPtr = &integralImagesGpuMem[imageHeight
         * imageWidth * loadImageIdx];

   cv::gpu::GpuMat gpuIntegralImage(imageHeight, imageWidth, CV_32S,
         integralImageGpuPtr);

   cv::Mat imageReadMat = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

   if (imageReadMat.empty())
   {
      std::cout << "Error: File:" << fileName << " is broken" << std::endl;
      fileError = true;
   }

   imageReadMat.copyTo(image);
   images.push_back(image);

   gpuImage.upload(image);
   gpuImages.push_back(gpuImage);
   const cv::gpu::PtrStepSz<uchar> imageMatPtr = gpuImages.back();
   imagePtrsVector.push_back(imageMatPtr);
   return fileError;
}

bool loadImages(
      int32_t * integralImagesGpuMem,
      const uint32_t maxImages, // maximum images to load into memory
      const uint32_t maxStagedImages, // maximum images for a stage
      const std::vector<std::string> & imageFileNames, // all file names
      const std::vector<uint32_t> & falsePositiveImages, // file indexes of false positive images
      const uint32_t positiveImageCount, // count of positive images
      const uint32_t imageWidth,
      const uint32_t imageHeight,
      uint32_t & falseImageStartIdx, // index of unloaded file
      std::vector<uint32_t> & availableIntegralImages, // indexes to images in memory
      std::vector<uint32_t> & loadedImageIdx, // indexes to file names
      uint32_t * availableIntegralImagesGpu,
      uint32_t & loadedImagesCount)
{
   assert(integralImagesGpuMem);
   assert(availableIntegralImagesGpu);
   availableIntegralImages.clear();
   loadedImagesCount = 0;
   loadedImageIdx.clear();
   loadedImageIdx.reserve(maxImages);

   size_t mem_tot_0 = 0;
   size_t mem_free_0 = 0;

   std::vector<cv::gpu::PtrStepSz<uchar> > imagePtrsVector;
   cv::gpu::PtrStepSz<uchar> * gpuImagesPtr;
   std::vector<cv::Mat> images;
   std::vector<cv::gpu::GpuMat> gpuImages;

   uchar * imagesMem = NULL;
   imagesMem =
         new uchar[maxImages * imageHeight * imageWidth];

   uchar * imagesGpuMem = NULL;
   CUDA_CHECK_RETURN(
         cudaMalloc((void** )&imagesGpuMem,
               maxImages * imageHeight * imageWidth));

   bool fileError = false;
   uint32_t loadImageIdx = 0;

   // load first positive images
   for (uint32_t i = 0; i < imageFileNames.size() && i < positiveImageCount && i < maxImages; ++i)
   {
      const std::string & fileName = imageFileNames[i];

      fileError |= loadImage(
            fileName,
            loadImageIdx,
            imageWidth, imageHeight,
            imagesMem, imagesGpuMem,
            integralImagesGpuMem,
            images, gpuImages,
            imagePtrsVector);

      if (loadImageIdx < maxStagedImages)
      {
         availableIntegralImages.push_back(loadImageIdx);
      }

      // index to loaded files
      loadedImageIdx.push_back(i);
      loadImageIdx++;
   }

   // load falsePositive images
   for (uint32_t i = 0; i < falsePositiveImages.size() && loadImageIdx < maxImages; ++i)
   {
      const uint32_t fileIdx = falsePositiveImages[i];
      const std::string & fileName = imageFileNames[fileIdx];

      fileError |= loadImage(
            fileName,
            loadImageIdx,
            imageWidth, imageHeight,
            imagesMem, imagesGpuMem,
            integralImagesGpuMem,
            images, gpuImages,
            imagePtrsVector);

      if (loadImageIdx < maxStagedImages)
      {
         availableIntegralImages.push_back(loadImageIdx);
      }

      loadedImageIdx.push_back(fileIdx);
      loadImageIdx++;
   }

   // fill up with unprocessed images
   for (; falseImageStartIdx < imageFileNames.size() && loadImageIdx < maxImages;)
   {
      const std::string & fileName = imageFileNames[falseImageStartIdx];

      fileError |= loadImage(
            fileName,
            loadImageIdx,
            imageWidth, imageHeight,
            imagesMem, imagesGpuMem,
            integralImagesGpuMem,
            images, gpuImages,
            imagePtrsVector);

      if (loadImageIdx < maxStagedImages)
      {
         availableIntegralImages.push_back(loadImageIdx);
      }

      loadedImageIdx.push_back(falseImageStartIdx);
      loadImageIdx++;
      falseImageStartIdx++;
   }

   if (fileError)
   {
      return 1;
   }

   const uint32_t curCountImages = availableIntegralImages.size();

   std::cout << "Count negative images - stage:" << curCountImages - positiveImageCount
         << std::endl;

   std::cout << "Count Images to be learned in this stage:" << curCountImages << std::endl;


   CUDA_CHECK_RETURN(
         cudaMalloc((void** )&gpuImagesPtr,
               loadImageIdx * sizeof(cv::gpu::PtrStepSz<uchar>)));

   CUDA_CHECK_RETURN(
         cudaMemcpy(gpuImagesPtr, &imagePtrsVector[0],
               loadImageIdx * sizeof(cv::gpu::PtrStepSz<uchar>),
               cudaMemcpyHostToDevice));

#ifdef DEBUG
   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << "Debug: Calc integral images:" << loadImageIdx << std::endl;
   std::cout << "Debug: Free: " << mem_free_0 << " Mem total: " << mem_tot_0
         << std::endl;
#endif

   getIntegralImage<<<(loadImageIdx + WORK_SIZE - 1) / WORK_SIZE, WORK_SIZE>>>(
         loadImageIdx, gpuImagesPtr, integralImagesGpuMem);

   CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
   CUDA_CHECK_RETURN(cudaGetLastError());

#ifdef DEBUG
   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << "Debug: Calc integral images download Done!:" << std::endl;
   std::cout << "Debug: Free: " << mem_free_0 << " Mem total: " << mem_tot_0
         << std::endl;
#endif

   //
   // we don't need the original images anymore - keep the integral images in memory
   //
   std::vector<cv::Mat>().swap(images);
   std::vector<cv::gpu::GpuMat>().swap(gpuImages);

   std::vector<cv::gpu::PtrStepSz<uchar> >().swap(imagePtrsVector);

   delete[] imagesMem;
   CUDA_CHECK_RETURN(cudaFree(gpuImagesPtr));
   CUDA_CHECK_RETURN(cudaFree(imagesGpuMem));

   // copy the integral images indexes to gpu
   CUDA_CHECK_RETURN(
         cudaMemcpy(availableIntegralImagesGpu, &availableIntegralImages[0],
               curCountImages * sizeof(uint32_t), cudaMemcpyHostToDevice));


#ifdef DEBUG
   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << "Debug: Cleared original images" << std::endl;
   std::cout << "Debug: Free memory: " << mem_free_0 << " Mem total: "
         << mem_tot_0 << std::endl;
#endif
   loadedImagesCount = loadImageIdx;
   return true;
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
   const uint32_t ratioX = 2;
   const uint32_t ratioY = 2;
   const double classifierScale = 1.25;
   const uint32_t maxLoadedImages = 40000;

   const uint32_t stages = 10;
   const double f = 0.3;  // false positive rate per layer
   const double d = 0.99; // minimum acceptable detection rate per layer
   const double fTarget = pow(f, stages); // overall false positive rate

   const uint32_t factorFirstStageImages = 4; // factor with positive images to learn the first stage
   const uint32_t factorImagesPerStage   = 2; // factor with positive images to learn the following stages

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

   cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 300);

   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << "Free memory:" << mem_free_0 << " Mem total: " << mem_tot_0
         << std::endl;

   // load images
   std::vector<std::string> fileNamesPos;
   std::vector<std::string> fileNamesNeg;
   std::vector<std::string> fileNames;

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

   const uint32_t maxAvailableImages = fileNames.size();
   const uint32_t maxImagesFirstStage = fileNamesPos.size() * factorFirstStageImages; // images to be learned at the first stage
   const uint32_t maxImagesPerStage = fileNamesPos.size() * factorImagesPerStage;     // images for learning at the following stages
   const uint32_t imagesToLoad = (maxLoadedImages > maxAvailableImages) ? maxAvailableImages : maxLoadedImages;
   uint32_t curCountImages = maxImagesFirstStage;

   // if not enough images available as need for the first
   // stage, reduce the image count to the available number.
   if (maxImagesFirstStage > imagesToLoad)
   {
      std::cout << "Less then " << maxImagesFirstStage << " where loaded:" << maxLoadedImages << std::endl;
      curCountImages = imagesToLoad;
   }

   assert(imagesToLoad > 0);

   int32_t * integralImagesGpuMem = NULL;

   CUDA_CHECK_RETURN(
         cudaMalloc((void** )&integralImagesGpuMem,
               imagesToLoad * sizeof(int32_t) * imageHeight * imageWidth));

   // pointer to integral images (index)
   std::vector<uint32_t> availableIntegralImages;
   std::vector<uint32_t> falsePositiveImages; // index to file names of false positive images
   std::vector<uint32_t> loadedImageIdx; // indexes to file names
   availableIntegralImages.reserve(curCountImages);

   uint32_t * availableIntegralImagesGpu = NULL;

   CUDA_CHECK_RETURN(
         cudaMalloc((void** )&availableIntegralImagesGpu,
               curCountImages * sizeof(uint32_t)));

   std::cout << "Count positive images:" << fileNamesPos.size()
         << std::endl;
   const uint32_t countPosImages = fileNamesPos.size();
   std::cout << "Count all negative images:" << maxAvailableImages - countPosImages
         << std::endl;

   std::cout << "Count Total Images to be learned:" << maxAvailableImages << std::endl;
   std::cout << "Ratio X:" << ratioX << " Y:" << ratioY << std::endl;

   // remove filenames
   std::vector<std::string>().swap(fileNamesPos);
   std::vector<std::string>().swap(fileNamesNeg);

   uint32_t falseImageStartIdx = countPosImages;
   uint32_t loadedImagesCount = 0;

   loadImages(
         integralImagesGpuMem,
         imagesToLoad,
         curCountImages,
         fileNames,
         falsePositiveImages,
         countPosImages,
         imageWidth,
         imageHeight,
         falseImageStartIdx,
         availableIntegralImages,
         loadedImageIdx,
         availableIntegralImagesGpu,
         loadedImagesCount);

   #ifdef DEBUG
   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << "Debug: Reserved space for images" << std::endl;
   std::cout << "Debug: Free: " << mem_free_0 << " Mem total: " << mem_tot_0
         << std::endl;
#endif

   /*
    cv::Mat displayImage;
    gpuIntegralImages[1].download(displayImage);
    Image::displayImageFalseColor(displayImage);
    return 0;
    */

   // init weight for all images
   double * imageWeightsPtr = new double[curCountImages];
   double * gpuImageWeightsPtr = NULL;

   CUDA_CHECK_RETURN(
         cudaMalloc((void ** )&gpuImageWeightsPtr,
               curCountImages * sizeof(double)));

   double initWeight = 1.0 / curCountImages;

   for (uint32_t i = 0; i < curCountImages; ++i)
   {
      imageWeightsPtr[i] = initWeight;
   }

   cudaMemcpy(gpuImageWeightsPtr, imageWeightsPtr, curCountImages * sizeof(double),
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
      falsePositiveIdx.reserve(curCountImages - countPosImages);

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

         adaBoost(
               roundIdx,
               curCountImages, countPosImages,
               pixelCount,
               ratioX,
               ratioY,
               imageWidth, imageHeight,
               featureTypes,
               integralImagesGpuMem,
               availableIntegralImagesGpu,
               imageWeightsPtr,
               gpuImageWeightsPtr,
               start, stop,
               classifierStage.stagedClassifier,
               results,
               gpuResults,
               newClassifierCount,
               outputfile);

         for (uint32_t i = classifierStage.stagedClassifier.size()
               - newClassifierCount;
               i < classifierStage.stagedClassifier.size(); ++i)
         {
            updateImageWeights(curCountImages, countPosImages,
                  integralImagesGpuMem, availableIntegralImagesGpu, imageWidth,
                  imageHeight, featureTypes,
                  classifierStage.stagedClassifier[i], imageWeightsPtr,
                  classifierStage.betas, featureValuesPtrs);

            cudaMemcpy(gpuImageWeightsPtr, imageWeightsPtr,
                  curCountImages * sizeof(double), cudaMemcpyHostToDevice);
         }

         if (newClassifierCount == 0)
         {
#ifdef DEBUG
            std::cout << "Debug: No new Classifier found" << std::endl;
#endif
            break;
         }

         evaluateClassifier(curCountImages, countPosImages,
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

      // create a new set of positive images and false positive images
      availableIntegralImages.clear();
      availableIntegralImages.reserve(maxImagesPerStage);

      for (uint32_t i = 0; i < countPosImages; ++i)
      {
         availableIntegralImages.push_back(i);
      }

      // add the false positive images to the set of images
      // that shall be used for learning in the next round

      uint32_t falsePositivesCount = 0;
      bool falsePositiveImagesDetectionDone = false;

      do
      {
#ifdef DEBUG
         std::cout << "Debug: Find false positives!"
               << std::endl;
#endif
         //use the generated classifier to detect all false positives on the complete set
         const uint32_t maxNegativeImages = loadedImagesCount - countPosImages;
         bool * falsePositives = new bool[maxNegativeImages];
         falsePositivesCount = 0;

         falsePositiveImages.clear();
         falsePositiveImages.reserve(maxImagesPerStage - countPosImages);

         Classifier::detectStrongClassifierOnImageSet(
               classifierStages,
               featureTypes,
               integralImagesGpuMem,
               countPosImages,
               maxNegativeImages,
               imageWidth,
               imageHeight,
               falsePositives
               );

         falsePositiveImagesDetectionDone = true;

         std::cout << "False Positive Images Indexes:";

         // fill up available images with false positives
         for (uint32_t falsePositiveIdxIter = 0;
               (falsePositiveIdxIter < maxNegativeImages) && (availableIntegralImages.size() <= maxImagesPerStage);
               ++falsePositiveIdxIter)
         {
            if (true == falsePositives[falsePositiveIdxIter])
            {
               falsePositivesCount++;
               availableIntegralImages.push_back(
                     falsePositiveIdxIter + countPosImages);
               const uint32_t fileImageIdx = loadedImageIdx[falsePositiveIdxIter + countPosImages];
               falsePositiveImages.push_back(fileImageIdx);
               std::cout << fileImageIdx << ",";
            }
         }

         std::cout << "Count:" << falsePositiveImages.size() << std::endl;

         delete[] falsePositives;

         if (availableIntegralImages.size() < maxImagesPerStage && falseImageStartIdx < maxAvailableImages)
         {
#ifdef DEBUG
            std::cout << "Debug: Reload images! falseImageStartIdx:" << falseImageStartIdx
                  << " falsePositivesCount:" << falsePositivesCount
                  << std::endl;
#endif
            // try to load more images if there are too less false positive images are available
            loadImages(
                  integralImagesGpuMem,
                  imagesToLoad,
                  curCountImages,
                  fileNames,
                  falsePositiveImages,
                  countPosImages,
                  imageWidth,
                  imageHeight,
                  falseImageStartIdx,
                  availableIntegralImages,
                  loadedImageIdx,
                  availableIntegralImagesGpu,
                  loadedImagesCount);

            falsePositiveImagesDetectionDone = false;
         }

      }
      while(availableIntegralImages.size() < maxImagesPerStage && falseImageStartIdx < maxAvailableImages && falsePositiveImagesDetectionDone == false);

      std::cout << "False Positive images detected (in loaded images):" << loadedImagesCount - countPosImages
                     << std::endl;

      std::cout << "False Positive images detected (in current stage):" << availableIntegralImages.size() - countPosImages
                     << std::endl;

      if (availableIntegralImages.size() <= countPosImages)
      {
#ifdef DEBUG
         std::cout << "Debug: No false positive images are left. Finishing!"
               << std::endl;
#endif
         break;
      }

      curCountImages = availableIntegralImages.size();

      CUDA_CHECK_RETURN(
            cudaMemcpy(availableIntegralImagesGpu, &availableIntegralImages[0],
                  curCountImages * sizeof(int32_t), cudaMemcpyHostToDevice));

#ifdef DEBUG
      std::cout << "Debug: Images left:" << curCountImages << std::endl;
#endif

      // init image weights
      double initWeight = 1.0 / curCountImages;

      for (uint32_t i = 0; i < curCountImages; ++i)
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

   outputfile << prettyClassifier.str();
   outputfile.close();

   CUDA_CHECK_RETURN(cudaFree(integralImagesGpuMem));
   CUDA_CHECK_RETURN(cudaFree(availableIntegralImagesGpu));
   delete[] imageWeightsPtr;
   CUDA_CHECK_RETURN(cudaFree(gpuImageWeightsPtr));
   //CUDA_CHECK_RETURN(cudaDeviceReset());
   return 0;
}
