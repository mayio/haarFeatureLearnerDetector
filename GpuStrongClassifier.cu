/*
 *
 *  Created on: May 17, 2017
 *      Author: Mario Lüder
 *
 */


#include "Classifier.cuh"
#include "GpuStrongClassifier.cuh"
#include "defines.cuh"

GpuStrongClassifier::GpuStrongClassifier(const std::vector<Classifier::Stage> & stagedClassifer)
: mStagesCount(stagedClassifer.size())
, mStages(NULL)
{
   CUDA_CHECK_RETURN(
         cudaMalloc(&mStages, sizeof(Stage) * mStagesCount));

   for (uint32_t i = 0; i < mStagesCount; ++i)
   {
      const Classifier::Stage & stage = stagedClassifer[i];
      Stage gpuStage;

      // create space for betas
      double * gpuBetaPtr;

      CUDA_CHECK_RETURN(
            cudaMalloc(&gpuBetaPtr, sizeof(double) * stage.betas.size()));
      CUDA_CHECK_RETURN(
            cudaMemcpy(
                  gpuBetaPtr,
                  &stage.betas[0],
                  sizeof(double) * stage.betas.size(),
                  cudaMemcpyHostToDevice));

      Classifier::SelectionResult * gpuSelectionResultPtr = NULL;

      CUDA_CHECK_RETURN(
            cudaMalloc(
                  &gpuSelectionResultPtr,
                  sizeof(Classifier::SelectionResult) * stage.stagedClassifier.size()));

      CUDA_CHECK_RETURN(
            cudaMemcpy(
                  gpuSelectionResultPtr,
                  &stage.stagedClassifier[0],
                  sizeof(Classifier::SelectionResult) * stage.stagedClassifier.size(),
                  cudaMemcpyHostToDevice));

      gpuStage.mBetas = gpuBetaPtr;
      gpuStage.mSelectionResults = gpuSelectionResultPtr;
      gpuStage.mClassifierCount = stage.stagedClassifier.size();
      gpuStage.mStageThreshold = stage.stageThreshold;

      CUDA_CHECK_RETURN(
            cudaMemcpy(
                  &mStages[i],
                  &gpuStage,
                  sizeof(Stage),
                  cudaMemcpyHostToDevice));

   }
}

GpuStrongClassifier::~GpuStrongClassifier()
{
   for (uint32_t i = 0; i < mStagesCount; ++i)
   {
      Stage cpuStage;

      CUDA_CHECK_RETURN(
            cudaMemcpy(
                  &cpuStage,
                  &mStages[i],
                  sizeof(Stage),
                  cudaMemcpyDeviceToHost));

      CUDA_CHECK_RETURN(cudaFree(cpuStage.mBetas));
      CUDA_CHECK_RETURN(cudaFree(cpuStage.mSelectionResults));
   }

   cudaFree(mStages);
}
