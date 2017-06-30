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
         "[[[[[[[4,18,1],[4,18,-1]]],56,40,0.112921,-1791,1],[[[[35,14,1]],[[35,14,-1]]],22,25,0.160909,15873,-1],[[[[7,11,1]],[[7,11,-1]]],15,2,0.229393,1072,-1],[[[[22,7,1]],[[22,7,-1]]],0,50,0.241598,-4134,1],[[[[11,28,1],[11,28,-1],[11,28,1]]],13,36,0.22637,39361,1]]," +
         "[0.127295,0.191766,0.297678,0.318562,0.292607]," +
         "2.37286]," +
         "[[[[[[11,28,1],[11,28,-1],[11,28,1]]],3,35,0.0920388,44285,1],[[[[35,4,1]],[[35,4,-1]]],18,56,0.0979379,-6207,1],[[[[4,14,1],[4,14,-1]]],56,42,0.225626,-634,1],[[[[3,14,1],[3,14,-1]]],0,44,0.271026,303,-1],[[[[28,9,1]],[[28,9,-1]]],29,38,0.302488,5531,-1],[[[[28,9,1]],[[28,9,-1]]],32,46,0.290169,-8578,1],[[[[2,28,1],[2,28,-1],[2,28,1]]],25,5,0.261227,3083,-1],[[[[11,14,1],[11,14,-1]]],1,9,0.262591,2186,-1],[[[[35,9,1]],[[35,9,-1]]],12,8,0.291488,4355,-1],[[[[35,11,1]],[[35,11,-1]]],19,42,0.330793,-22381,1],[[[[2,7,1],[2,7,-1]]],22,57,0.284522,-26,-1],[[[[35,11,1]],[[35,11,-1]]],12,41,0.265543,14866,-1],[[[[2,35,1],[2,35,-1],[2,35,1]]],40,1,0.305008,5727,1],[[[[7,23,1]],[[7,23,-1]]],11,7,0.312512,6835,-1],[[[[11,6,1],[11,6,-1]]],11,14,0.337382,-723,1],[[[[2,7,1],[2,7,-1]]],57,57,0.341284,12,1],[[[[35,6,1]],[[35,6,-1]]],7,17,0.322762,-4981,1]]," +
         "[0.101369,0.108571,0.291366,0.37179,0.433667,0.408786,0.353595,0.356099,0.411409,0.494305,0.397668,0.361551,0.438865,0.454571,0.509164,0.518104,0.476586]," +
         "8.47287]," +
         "[[[[[[4,9,1],[4,9,-1]]],18,55,0.0164101,719,1],[[[[6,14,1],[6,14,-1]]],52,43,0.180468,-4192,1],[[[[2,7,1],[2,7,-1],[2,7,1]]],53,48,0.276319,329,1],[[[[4,18,1],[4,18,-1]]],0,44,0.289576,456,-1],[[[[4,7,1],[4,7,-1]]],3,17,0.302191,249,-1],[[[[4,7,1],[4,7,-1]]],17,57,0.271353,-158,-1],[[[[22,3,1]],[[22,3,-1]]],18,5,0.287661,-417,1],[[[[11,3,1]],[[11,3,-1]]],0,43,0.316637,84,-1],[[[[14,11,1]],[[14,11,-1]]],36,23,0.312994,1471,-1],[[[[3,9,1],[3,9,-1]]],25,55,0.32938,-101,-1],[[[[35,6,1]],[[35,6,-1]]],15,3,0.290933,4828,-1],[[[[4,18,1],[4,18,-1]]],56,46,0.31619,-124,1],[[[[9,7,1]],[[9,7,-1]]],42,48,0.32906,-2360,1],[[[[4,9,1],[4,9,-1]]],22,49,0.326384,200,1],[[[[4,7,1],[4,7,-1]]],53,16,0.320589,486,-1],[[[[4,22,1],[4,22,-1]]],33,15,0.307745,-53,-1],[[[[7,7,1]],[[7,7,-1]]],20,10,0.339455,549,-1],[[[[2,6,1],[2,6,-1]]],22,42,0.335931,-11,-1],[[[[28,4,1]],[[28,4,-1]]],4,44,0.318995,561,-1],[[[[11,14,1],[11,14,-1]]],11,28,0.328759,-1130,1]]," +
         "[0.0166839,0.220208,0.381824,0.40761,0.433058,0.372407,0.403826,0.46335,0.455591,0.491158,0.410305,0.462395,0.490447,0.484526,0.471862,0.444554,0.513902,0.505867,0.468418,0.489778]," +
         "11.1484]," +
         "[[[[[[18,3,1]],[[18,3,-1]]],0,12,0.00265346,2456,1],[[[[7,3,1]],[[7,3,-1]]],6,20,0.000466991,1145,1]]," +
         "[0.00266052,0.00046721]," +
         "13.597]";
         //"13.598]";


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
   /*
   classifierScales.push_back(0.5);
   classifierScales.push_back(0.75);
   classifierScales.push_back(1.0);
   classifierScales.push_back(1.1);
   classifierScales.push_back(1.2);
   classifierScales.push_back(1.4);
   */
   classifierScales.push_back(1.6);

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
