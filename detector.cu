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

// load image includes
#include <opencv2/core/core.hpp>
#include <opencv2/core/gpumat.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_profiler_api.h>

#include "utilities.cuh"
#include "defines.cuh"

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void)
{
   // Original
   // std::string strongClassifierStr = "[[[[[[[49,22,1]],[[49,22,-1]]],18,0,0.0332471,-90686,1],[[[[12,36,1],[12,36,-1]]],0,12,0.0991827,-9593,1]],[0.0343905,0.110103],2.20634],[[[[[[16,66,1],[16,66,-1]]],57,20,0.108966,29552,-1],[[[[14,16,1]],[[14,16,-1]]],72,64,0.160855,3968,-1],[[[[26,4,1]],[[26,4,-1]]],33,100,0.183328,1901,-1],[[[[29,36,1],[29,36,-1]]],33,8,0.1819,25689,-1],[[[[4,6,1],[4,6,-1]]],84,0,0.169853,44211,1]],[0.122292,0.191689,0.224481,0.222345,0.204606],5.09883],[[[[[[26, 29, 1]], [[26, 29, -1]]], 18, 32, 0.0191388, 34104, 1],[[[[26, 3, 1]], [[26, 3, -1]], [[26, 3, 1]]], 21, 52, 0.0207317, 8115, -1]], [0.0195122, 0.0211706], 6.79186]]";

   //std::string strongClassifierStr = "[[[[[[[49,22,1]],[[49,22,-1]]],18,0,0.0332471,-90686,1],[[[[12,36,1],[12,36,-1]]],0,12,0.0991827,-9593,1]],[0.0343905,0.110103],2.20634],[[[[[[16,66,1],[16,66,-1]]],57,20,0.108966,29552,-1],[[[[14,16,1]],[[14,16,-1]]],72,64,0.160855,3968,-1],[[[[26,4,1]],[[26,4,-1]]],33,100,0.183328,1901,-1],[[[[29,36,1],[29,36,-1]]],33,8,0.1819,25689,-1],[[[[4,6,1],[4,6,-1]]],84,0,0.169853,44211,1]],[0.122292,0.191689,0.224481,0.222345,0.204606],5.09883],[[[[[[26, 29, 1]], [[26, 29, -1]]], 18, 32, 0.0191388, 34104, 1],[[[[26, 3, 1]], [[26, 3, -1]], [[26, 3, 1]]], 21, 52, 0.0207317, 8115, -1]], [0.0195122, 0.0211706], 7.78186]]";
   /*
   [[[[[[[49,22,1]],[[49,22,-1]]],18,0,0.0332471,-90686,1],[[[[22,8,1],[22,8,-1]]],48,104,0.0922494,-38536,1],[[[[8,26,1],[8,26,-1]]],0,20,0.144086,-6474,1],[[[[10,22,1]],[[10,22,-1]],[[10,22,1]]],0,32,0.160434,18853,1],[[[[26,4,1]],[[26,4,-1]]],30,100,0.154933,1657,-1]],
   [0.0343905,0.101624,0.168342,0.191091,0.183338],
   3.35143],
   [[[[[[49,22,1]],[[49,22,-1]]],6,0,0.176727,-40517,1],[[[[12,66,1],[12,66,-1]]],66,24,0.187417,21526,-1],[[[[14,16,1]],[[14,16,-1]]],75,64,0.194728,3836,-1],[[[[12,36,1],[12,36,-1]]],0,56,0.23373,-8190,1],[[[[22,66,1],[22,66,-1]]],39,0,0.228247,21777,-1],[[[[49,6,1]],[[49,6,-1]]],9,0,0.228516,-7595,1],[[[[49,6,1]],[[49,6,-1]]],18,52,0.236501,-3840,1],[[[[26,12,1]],[[26,12,-1]]],0,64,0.225154,2591,-1],[[[[49,3,1]],[[49,3,-1]]],21,48,0.265981,-921,1],[[[[6,3,1]],[[6,3,-1]]],39,104,0.264232,57,-1],[[[[14,3,1]],[[14,3,-1]]],6,104,0.265039,-136,1],[[[[10,22,1]],[[10,22,-1]],[[10,22,1]]],78,36,0.252234,16527,1],[[[[2,6,1],[2,6,-1]]],69,100,0.242955,396,-1]],
   [0.214664,0.230644,0.241816,0.305022,0.295752,0.296203,0.309759,0.290579,0.362362,0.359124,0.360616,0.337317,0.320925],
   7.37621],
   [[[[[[49,4,1]],[[49,4,-1]],[[49,4,1]]],42,0,0.0261283,29220,1],[[[[12,26,1],[12,26,-1],[12,26,1]]],39,32,0.0710643,28378,-1],[[[[2,26,1],[2,26,-1]]],72,12,0.11069,52,-1],[[[[29,36,1],[29,36,-1]]],24,48,0.104579,1347,-1],[[[[8,36,1],[8,36,-1]]],24,20,0.0969391,1184,1]],
   [0.0268293,0.0765008,0.124468,0.116793,0.107345],
   8.33607]
   ]
   */


   /* faces
   std::string strongClassifierStr = std::string() +
         "[[[[[[[49,22,1]],[[49,22,-1]]],18,0,0.0332471,-90686,1],[[[[12,36,1],[12,36,-1]]],0,12,0.0991827,-9593,1],[[[[16,6,1],[16,6,-1]]],57,84,0.148353,2177,-1]]," +
         "[0.0343905,0.110103,0.174196]," +
         "1.74758]," +
         "[[[[[[8,12,1]],[[8,12,-1]]],84,64,0.0523191,-7807,1],[[[[14,16,1]],[[14,16,-1]]],33,12,0.0340082,-27205,1],[[[[22,36,1],[22,36,-1]]],45,12,0.114434,21409,-1],[[[[14,22,1]],[[14,22,-1]]],12,0,0.189615,-7194,1],[[[[26,4,1]],[[26,4,-1]]],30,100,0.165921,1563,-1],[[[[6,12,1]],[[6,12,-1]]],81,88,0.183366,66521,1]]," +
         "[0.0552076,0.0352054,0.129221,0.233982,0.198927,0.224539]," +
         "5.84288]," +
         "[[[[[[12,26,1],[12,26,-1],[12,26,1]]],36,76,0.0178653,22526,-1],[[[[2,10,1],[2,10,-1]]],3,20,0.0707651,-591,1],[[[[12,26,1],[12,26,-1]]],0,72,0.173719,-2579,1],[[[[49,16,1]],[[49,16,-1]]],27,0,0.175738,-24893,1],[[[[10,4,1]],[[10,4,-1]]],72,80,0.21342,181,-1],[[[[29,49,1],[29,49,-1]]],30,56,0.221047,23861,-1],[[[[6,8,1]],[[6,8,-1]],[[6,8,1]]],0,8,0.209865,4279,1],[[[[6,6,1],[6,6,-1]]],12,84,0.220425,-470,1],[[[[6,6,1]],[[6,6,-1]]],6,84,0.210801,-16,-1],[[[[49,6,1]],[[49,6,-1]]],15,96,0.222737,5228,-1],[[[[8,26,1],[8,26,-1]]],60,8,0.230551,914,-1]]," +
         "[0.0181903,0.0761542,0.210242,0.213206,0.271327,0.283774,0.265606,0.282749,0.267108,0.286566,0.299632]," +
         "10.5988]" +
         "]";
   */

   const std::string strongClassifierStr = std::string() +
         "[[[[[[[35,11,1]],[[35,11,-1]]],12,42,0.0640572,-13744,1],[[[[4,14,1],[4,14,-1]]],48,38,0.142968,-600,1],[[[[11,18,1],[11,18,-1]]],0,40,0.204615,2112,-1]]," +
         "[0.0684414,0.166818,0.257252]," +
         "1.3577]," +
         "[[[[[[7,7,1],[7,7,-1]]],18,54,0.144448,-272,-1],[[[[11,9,1]],[[11,9,-1]]],40,42,0.109374,-4519,1],[[[[6,9,1],[6,9,-1]]],6,40,0.169851,1339,-1],[[[[11,18,1],[11,18,-1]]],42,36,0.224198,-3089,1],[[[[7,6,1],[7,6,-1]]],20,20,0.277176,-164,1],[[[[22,3,1]],[[22,3,-1]]],22,16,0.290301,559,-1],[[[[11,11,1]],[[11,11,-1]]],28,42,0.289367,-2008,1],[[[[6,11,1]],[[6,11,-1]]],10,30,0.30134,620,-1],[[[[35,11,1]],[[35,11,-1]]],16,0,0.308367,-2129,1],[[[[2,22,1],[2,22,-1]]],36,42,0.316415,-33,-1],[[[[7,14,1],[7,14,-1]]],44,26,0.302584,1773,-1],[[[[2,14,1],[2,14,-1]]],22,48,0.315044,24,1],[[[[11,6,1],[11,6,-1]]],36,22,0.305314,-1035,1],[[[[7,28,1],[7,28,-1]]],20,20,0.330921,-112,1],[[[[4,7,1],[4,7,-1]]],48,42,0.353105,-200,1],[[[[3,14,1],[3,14,-1]]],30,42,0.347457,-82,-1],[[[[18,4,1]],[[18,4,-1]]],30,44,0.309401,-2275,1]]," +
         "[0.168836,0.122805,0.204603,0.288989,0.383462,0.409047,0.407197,0.431311,0.445853,0.462875,0.433864,0.459947,0.439499,0.494591,0.545846,0.532467,0.448018]," +
         "7.8044]," +
         "[[[[[[3,18,1],[3,18,-1]]],26,42,0.0864314,-274,-1],[[[[11,7,1],[11,7,-1]]],0,42,0.208462,3521,-1],[[[[9,7,1]],[[9,7,-1]]],12,44,0.25258,-1571,1],[[[[6,11,1],[6,11,-1]]],46,40,0.280878,-1835,1],[[[[35,6,1]],[[35,6,-1]]],8,16,0.332523,2286,-1],[[[[9,6,1],[9,6,-1]]],26,48,0.327122,335,1],[[[[22,6,1]],[[22,6,-1]]],22,20,0.297943,-2521,1],[[[[7,28,1],[7,28,-1]]],28,32,0.319439,-812,-1],[[[[14,2,1]],[[14,2,-1]],[[14,2,1]]],26,42,0.317712,314,1],[[[[14,22,1],[14,22,-1]]],0,38,0.322123,1424,-1],[[[[18,4,1]],[[18,4,-1]]],24,42,0.335708,990,-1],[[[[6,7,1],[6,7,-1]]],24,46,0.310956,240,1],[[[[9,7,1]],[[9,7,-1]]],42,10,0.335895,-383,1],[[[[2,11,1],[2,11,-1]]],36,46,0.35437,26,1],[[[[28,7,1]],[[28,7,-1]]],18,40,0.316842,-9353,1],[[[[7,11,1]],[[7,11,-1]]],46,30,0.335667,175,-1],[[[[4,14,1],[4,14,-1]]],32,20,0.349175,42,-1],[[[[7,3,1]],[[7,3,-1]]],14,28,0.356909,45,-1],[[[[3,9,1],[3,9,-1]]],12,20,0.355415,107,-1],[[[[3,11,1],[3,11,-1]]],38,48,0.343201,41,1],[[[[4,9,1],[4,9,-1]]],42,14,0.362405,136,-1],[[[[3,6,1],[3,6,-1]]],48,56,0.357153,-43,-1],[[[[4,7,1],[4,7,-1]]],42,18,0.327319,-420,1],[[[[2,6,1],[2,6,-1]]],22,36,0.352435,9,1],[[[[3,18,1],[3,18,-1]]],12,26,0.357045,-197,1],[[[[7,11,1]],[[7,11,-1]]],28,42,0.34402,-396,1],[[[[3,7,1],[3,7,-1]]],10,36,0.346336,196,-1],[[[[4,14,1],[4,14,-1]]],0,50,0.346424,-81,-1],[[[[7,7,1],[7,7,-1],[7,7,1]]],18,34,0.35411,6881,-1],[[[[2,11,1],[2,11,-1]]],32,40,0.328788,16,1]]," +
         "[0.0946085,0.263364,0.337936,0.390584,0.49818,0.486155,0.424386,0.469375,0.465657,0.475193,0.505362,0.451286,0.505787,0.548874,0.463789,0.505269,0.536511,0.55499,0.551385,0.522535,0.568393,0.555579,0.48659,0.544245,0.555319,0.524436,0.529839,0.530045,0.548252,0.489842]," +
         "12.3428]," +
         "[[[[[[7,22,1],[7,22,-1],[7,22,1]]],14,42,0.00614771,2560,-1],[[[[7,35,1],[7,35,-1]]],28,20,0.0868318,-2084,-1],[[[[6,11,1]],[[6,11,-1]],[[6,11,1]]],46,18,0.0581834,7942,1],[[[[4,9,1],[4,9,-1]]],24,34,0.0547779,151,1]]," +
         "[0.00618574,0.0950885,0.0617778,0.0579524]," +
         "10.2227]" +
         "]";



   // test face
   //const std::string imageFileName = "/mnt/project-disk/src/ObjectRecognition/data/facesTraining/tutorial-haartraining/data/CMU-MIT_Face_Test_Set/newtest/ew-courtney-david.png";

   // test car
   const std::string imageFileName = "/mnt/project-disk/src/ObjectRecognition/data/cars/TheKITTIVision/testing/image_2/000006.png";

   deviceSetup();

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   Image image;
   Image::fromFile(imageFileName, image);

   std::vector<Classifier::Stage> strongClassifier;
   FeatureTypes featureTypes;
   Classifier::fromResult(strongClassifierStr, strongClassifier, featureTypes);

   std::vector<Classifier::ClassificationResult> results;

   // define all scales for a strong classifier
   std::vector<double> classifierScales;

   // classifierScales.push_back(0.5);
   // classifierScales.push_back(0.75);
   classifierScales.push_back(1.0);
   // classifierScales.push_back(1.1);
   // classifierScales.push_back(1.2);
   // classifierScales.push_back(1.4);
   // classifierScales.push_back(1.6);

   // use the defined scales to detect objects
   for (std::vector<double>::const_iterator classifierScalesIter = classifierScales.begin();
         classifierScalesIter != classifierScales.end();
         ++classifierScalesIter)
   {
      std::vector<Classifier::Stage> scaledStrongClassifier;
      FeatureTypes scaledFeatureTypes;
      Classifier::scaleStrongClassifier(*classifierScalesIter, strongClassifier, featureTypes, scaledStrongClassifier, scaledFeatureTypes);
      scaledFeatureTypes.generateClassifier(1.0, image.getWidth(), image.getHeight(), true);
      Classifier::detectStrongClassifier(scaledStrongClassifier, scaledFeatureTypes, image.getGpuIntegralImage(), results);

   }

   image.displayClassificationResult(results);

	return 0;
}
