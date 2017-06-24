/*
 *
 *  Created on: May 26, 2017
 *      Author: Mario Lüder
 *
 */


#ifndef GPUSTRONGCLASSIFIER_CUH_
#define GPUSTRONGCLASSIFIER_CUH_

#include <vector>

// forward declaration
class Classifier;

class GpuStrongClassifier
{
public:
   struct Stage
   {
      double * mBetas;
      Classifier::SelectionResult * mSelectionResults;
      uint32_t mClassifierCount;
      double mStageThreshold;
   };

   GpuStrongClassifier(const std::vector<Classifier::Stage> & stagedClassifer);
   virtual ~GpuStrongClassifier();

   const uint32_t mStagesCount;

   const Stage * getGpuStages() const {return mStages; }
private:
   Stage * mStages;
};

#endif /* GPUSTRONGCLASSIFIER_CUH_ */

