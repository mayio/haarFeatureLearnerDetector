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
         "[[[[[[[37,11,1]],[[37,11,-1]]],12,42,0.0637033,-16541,1],[[[[9,12,1],[9,12,-1]]],4,40,0.143831,2155,-1],[[[[6,19,1],[6,19,-1]]],46,38,0.234012,-1040,1],[[[[7,12,1],[7,12,-1]]],20,20,0.289658,-146,1],[[[[6,7,1],[6,7,-1]]],46,26,0.296355,632,-1]]," +
         "[0.0680375,0.167993,0.305504,0.407772,0.421172]," +
         "1.18579]," +
         "[[[[[[2,12,1],[2,12,-1]]],24,52,0.110073,-28,-1],[[[[37,7,1]],[[37,7,-1]]],12,44,0.0887522,-13006,1],[[[[9,9,1],[9,9,-1]]],44,40,0.181879,-2270,1],[[[[11,19,1],[11,19,-1]]],0,38,0.203371,3600,-1],[[[[6,11,1]],[[6,11,-1]]],26,42,0.26366,-1099,1],[[[[46,7,1]],[[46,7,-1]]],10,50,0.279737,-13429,1],[[[[7,12,1],[7,12,-1]]],20,20,0.282004,-146,1],[[[[46,9,1]],[[46,9,-1]]],12,2,0.306152,-1530,1],[[[[6,12,1],[6,12,-1]]],32,48,0.317257,233,1]]," +
         "[0.123687,0.0973963,0.222314,0.25529,0.358069,0.388381,0.392765,0.441238,0.464679]," +
         "4.60912]," +
         "[[[[[[23,4,1]],[[23,4,-1]]],18,16,0.314598,1180,-1],[[[[37,9,1]],[[37,9,-1]]],12,42,0.169708,-11548,1],[[[[6,19,1],[6,19,-1]]],6,40,0.214206,945,-1],[[[[6,15,1],[6,15,-1]]],46,40,0.264625,-1647,1],[[[[23,14,1]],[[23,14,-1]]],2,24,0.308027,1815,-1],[[[[2,12,1],[2,12,-1]]],24,52,0.34571,15,1],[[[[37,7,1]],[[37,7,-1]]],14,40,0.318646,-13009,1],[[[[9,12,1],[9,12,-1]]],24,40,0.314639,-730,-1],[[[[46,7,1]],[[46,7,-1]]],0,38,0.323005,7049,-1],[[[[6,6,1],[6,6,-1]]],28,46,0.324259,148,1],[[[[9,7,1],[9,7,-1]]],10,20,0.339982,542,-1],[[[[7,11,1]],[[7,11,-1]]],46,28,0.349027,424,-1],[[[[29,4,1]],[[29,4,-1]]],22,42,0.328483,-2367,1]]," +
         "[0.458997,0.204395,0.272599,0.35985,0.445143,0.528375,0.467666,0.459084,0.477115,0.479857,0.515109,0.536162,0.489165]," +
         "4.24679]," +
         "[[[[[[7,29,1],[7,29,-1]]],30,32,0.336631,-385,-1],[[[[6,9,1],[6,9,-1]]],6,40,0.226377,1599,-1],[[[[4,11,1]],[[4,11,-1]]],46,42,0.282765,-1723,1],[[[[14,7,1],[14,7,-1]]],36,42,0.301254,-2345,1],[[[[29,2,1]],[[29,2,-1]]],18,14,0.313999,-273,1],[[[[6,19,1],[6,19,-1]]],22,20,0.332625,-175,1],[[[[6,15,1],[6,15,-1]]],30,40,0.352378,232,1],[[[[19,3,1]],[[19,3,-1]]],18,24,0.326638,-529,1],[[[[6,12,1],[6,12,-1]]],0,48,0.349744,-226,-1],[[[[4,6,1],[4,6,-1]]],42,20,0.351629,-287,1],[[[[11,6,1],[11,6,-1]]],38,56,0.34938,-465,-1],[[[[4,15,1],[4,15,-1]]],48,32,0.351483,-1461,1],[[[[14,15,1],[14,15,-1]]],14,46,0.374114,-2604,-1],[[[[6,12,1],[6,12,-1]]],46,30,0.337285,955,-1],[[[[6,9,1],[6,9,-1]]],24,46,0.350289,231,1],[[[[29,2,1]],[[29,2,-1]],[[29,2,1]]],16,42,0.343238,778,1],[[[[9,19,1],[9,19,-1]]],4,42,0.367173,909,-1],[[[[15,6,1]],[[15,6,-1]]],24,18,0.366343,1010,-1],[[[[9,4,1],[9,4,-1]]],2,60,0.373107,290,1],[[[[23,3,1]],[[23,3,-1]]],18,14,0.347645,965,-1],[[[[4,19,1],[4,19,-1]]],28,42,0.37009,-261,-1],[[[[4,6,1],[4,6,-1]]],42,20,0.365143,274,-1]]," +
         "[0.507458,0.292619,0.394243,0.431135,0.457724,0.498408,0.54411,0.485085,0.537856,0.542328,0.536996,0.54198,0.597735,0.508943,0.539145,0.522621,0.58021,0.57814,0.595168,0.532908,0.587528,0.575158]," +
         "6.14831]," +
         "[[[[[[3,5,1],[3,5,-1]]],30,36,0.366036,32,1],[[[[6,12,1],[6,12,-1]]],46,40,0.257528,-2291,1],[[[[4,7,1]],[[4,7,-1]]],14,44,0.293155,-744,1],[[[[5,9,1]],[[5,9,-1]]],26,32,0.327902,1376,-1],[[[[11,12,1],[11,12,-1]]],0,34,0.341991,4611,-1],[[[[6,7,1]],[[6,7,-1]]],20,12,0.360899,366,-1],[[[[4,11,1]],[[4,11,-1]]],34,42,0.370884,-553,1],[[[[7,7,1],[7,7,-1]]],6,24,0.349424,-583,1],[[[[3,6,1],[3,6,-1]]],30,36,0.36388,-27,-1],[[[[3,6,1],[3,6,-1]]],12,22,0.357867,137,-1],[[[[46,11,1]],[[46,11,-1]]],6,0,0.371069,-1977,1],[[[[7,4,1],[7,4,-1]]],40,30,0.381094,-182,1],[[[[3,7,1],[3,7,-1]]],38,50,0.377854,28,1],[[[[37,6,1]],[[37,6,-1]]],14,40,0.354227,7645,-1],[[[[6,9,1],[6,9,-1]]],20,36,0.354142,128,1],[[[[23,4,1]],[[23,4,-1]]],24,44,0.37442,-2751,1],[[[[3,4,1],[3,4,-1]]],26,44,0.359185,-25,-1],[[[[15,3,1]],[[15,3,-1]]],28,46,0.372198,330,-1],[[[[4,4,1],[4,4,-1]]],48,60,0.364644,-42,-1],[[[[6,3,1]],[[6,3,-1]]],18,14,0.374577,-119,1],[[[[12,7,1]],[[12,7,-1]]],30,30,0.374294,378,-1],[[[[3,37,1],[3,37,-1]]],34,14,0.396341,-32,-1],[[[[4,6,1],[4,6,-1]]],46,24,0.384194,423,-1],[[[[9,5,1],[9,5,-1]]],46,42,0.373195,-232,1],[[[[14,29,1],[14,29,-1]]],4,24,0.389012,6201,-1],[[[[3,19,1],[3,19,-1]]],46,30,0.389558,99,-1],[[[[4,9,1],[4,9,-1]]],26,36,0.386639,75,1],[[[[19,3,1]],[[19,3,-1]]],26,14,0.383729,1234,-1]]," +
         "[0.577378,0.346853,0.414737,0.487878,0.519736,0.564699,0.589532,0.537099,0.572031,0.557311,0.59,0.615754,0.607339,0.548531,0.548327,0.598517,0.560512,0.592859,0.573921,0.598918,0.598195,0.656565,0.623887,0.595392,0.636694,0.638157,0.630362,0.622664]," +
         "6.75385]," +
         "[[[[[[3,12,1],[3,12,-1]]],0,44,0.358418,-53,-1],[[[[4,9,1],[4,9,-1]]],48,40,0.292702,-1030,1],[[[[6,7,1]],[[6,7,-1]]],12,46,0.325383,-1772,1],[[[[4,4,1],[4,4,-1]]],8,40,0.350177,265,-1],[[[[37,6,1]],[[37,6,-1]]],10,16,0.372089,1581,-1],[[[[4,6,1]],[[4,6,-1]]],12,36,0.372521,176,-1],[[[[29,2,1]],[[29,2,-1]]],20,46,0.378371,-375,1],[[[[3,6,1],[3,6,-1]]],34,48,0.363615,25,1],[[[[29,2,1]],[[29,2,-1]]],16,44,0.383787,-519,1],[[[[4,15,1],[4,15,-1]]],32,42,0.359615,-218,-1],[[[[19,3,1]],[[19,3,-1]]],22,44,0.371061,869,-1],[[[[9,18,1]],[[9,18,-1]]],42,18,0.364361,357,-1],[[[[4,3,1]],[[4,3,-1]]],38,16,0.388302,93,-1],[[[[6,12,1],[6,12,-1]]],30,20,0.388221,60,-1],[[[[4,23,1]],[[4,23,-1]]],60,8,0.392129,-1067,1],[[[[4,4,1],[4,4,-1]]],30,40,0.389709,45,1],[[[[23,2,1]],[[23,2,-1]]],22,16,0.385037,-316,1],[[[[7,9,1],[7,9,-1]]],6,44,0.376134,311,-1],[[[[7,3,1]],[[7,3,-1]]],14,28,0.390252,47,-1],[[[[12,4,1]],[[12,4,-1]]],22,14,0.393533,710,-1],[[[[6,6,1],[6,6,-1]]],42,58,0.393189,-146,-1],[[[[29,2,1]],[[29,2,-1]],[[29,2,1]]],20,42,0.3821,379,1],[[[[2,15,1],[2,15,-1]]],26,48,0.383666,-26,-1],[[[[46,3,1]],[[46,3,-1]]],0,42,0.386282,1240,-1],[[[[7,9,1]],[[7,9,-1]]],46,40,0.387686,-307,1],[[[[29,3,1]],[[29,3,-1]]],14,46,0.39242,723,-1],[[[[3,4,1],[3,4,-1]]],28,36,0.383637,25,1],[[[[23,3,1]],[[23,3,-1]]],22,14,0.39268,-349,1],[[[[4,29,1],[4,29,-1]]],22,20,0.395364,-149,1],[[[[14,9,1],[14,9,-1]]],18,34,0.403841,-565,-1],[[[[9,6,1]],[[9,6,-1]]],16,40,0.393095,-2354,1],[[[[7,9,1],[7,9,-1]]],44,42,0.389249,-407,1],[[[[2,15,1],[2,15,-1]]],48,28,0.394401,75,-1],[[[[3,29,1],[3,29,-1]]],58,34,0.388329,222,1],[[[[2,12,1],[2,12,-1]]],48,22,0.395636,-64,1],[[[[15,3,1]],[[15,3,-1]]],42,38,0.40081,67,-1],[[[[3,19,1],[3,19,-1]]],12,26,0.405555,-68,1],[[[[2,15,1],[2,15,-1]]],26,46,0.395996,27,1]]," +
         "[0.558646,0.41383,0.482323,0.53888,0.592581,0.593679,0.608676,0.571377,0.622817,0.561561,0.58998,0.573219,0.634793,0.634577,0.645087,0.638561,0.626115,0.602908,0.640021,0.648893,0.647959,0.618385,0.622497,0.629413,0.633148,0.645873,0.622421,0.64658,0.653888,0.677406,0.647704,0.637328,0.651259,0.634865,0.654631,0.668919,0.682241,0.655619]," +
         "8.30001]," +
         "[[[[[[3,12,1],[3,12,-1]]],10,28,0.395429,256,-1],[[[[11,4,1],[11,4,-1]]],0,44,0.323442,1565,-1],[[[[7,6,1]],[[7,6,-1]]],26,24,0.358018,-140,1],[[[[4,23,1],[4,23,-1]]],48,40,0.368201,-345,1],[[[[37,4,1]],[[37,4,-1]]],16,48,0.365094,-5205,1],[[[[6,23,1]],[[6,23,-1]]],26,8,0.388224,653,-1],[[[[15,7,1]],[[15,7,-1]]],14,40,0.370167,-4307,1],[[[[3,4,1],[3,4,-1]]],24,20,0.393338,-3,1],[[[[29,3,1]],[[29,3,-1]]],28,24,0.389752,-502,1],[[[[6,15,1],[6,15,-1]]],28,44,0.383547,338,1],[[[[29,3,1]],[[29,3,-1]]],18,12,0.381848,-1050,1],[[[[7,12,1],[7,12,-1]]],24,44,0.384256,-347,-1],[[[[29,3,1]],[[29,3,-1]]],18,12,0.388769,1184,-1],[[[[3,9,1],[3,9,-1]]],4,42,0.379634,163,1],[[[[6,12,1],[6,12,-1]]],6,24,0.402096,-1283,1],[[[[3,4,1],[3,4,-1]]],20,36,0.388747,21,1],[[[[2,6,1],[2,6,-1]]],14,20,0.401192,35,-1],[[[[23,11,1]],[[23,11,-1]]],24,0,0.403866,-1003,1],[[[[7,4,1],[7,4,-1]]],38,22,0.410301,-178,1],[[[[2,6,1],[2,6,-1]]],28,40,0.408001,-8,-1],[[[[23,2,1]],[[23,2,-1]]],18,44,0.396568,375,-1],[[[[4,29,1],[4,29,-1]]],26,24,0.391429,198,1],[[[[3,5,1],[3,5,-1],[3,5,1]]],42,42,0.406066,271,1],[[[[3,5,1],[3,5,-1]]],50,56,0.40617,-34,-1],[[[[19,2,1]],[[19,2,-1]]],22,44,0.393011,-313,1],[[[[4,11,1]],[[4,11,-1]]],56,40,0.397872,-1460,-1],[[[[3,9,1],[3,9,-1]]],48,34,0.400496,190,-1],[[[[11,19,1],[11,19,-1]]],18,12,0.391616,-4534,-1],[[[[23,3,1]],[[23,3,-1]]],20,46,0.39338,-2493,1],[[[[2,23,1],[2,23,-1]]],56,40,0.399842,48,1],[[[[4,6,1],[4,6,-1]]],44,22,0.403868,191,-1],[[[[14,15,1],[14,15,-1]]],0,44,0.412586,1298,-1],[[[[2,9,1],[2,9,-1]]],48,26,0.396568,-99,1],[[[[2,29,1],[2,29,-1]]],56,34,0.413008,-80,-1],[[[[12,4,1]],[[12,4,-1]]],30,14,0.403089,623,-1],[[[[5,11,1]],[[5,11,-1]]],0,2,0.418581,-130,1],[[[[5,6,1]],[[5,6,-1]]],40,26,0.419844,-10,-1],[[[[19,2,1]],[[19,2,-1]]],22,30,0.409686,-249,1],[[[[2,15,1],[2,15,-1]]],24,48,0.400867,36,1],[[[[23,2,1]],[[23,2,-1]]],20,16,0.406391,-418,1],[[[[6,23,1]],[[6,23,-1]]],26,12,0.405301,-1692,-1],[[[[46,9,1]],[[46,9,-1]],[[46,9,1]]],2,36,0.408024,78241,-1],[[[[19,23,1]],[[19,23,-1]]],10,14,0.400705,16737,1],[[[[3,5,1],[3,5,-1]]],12,24,0.408838,-145,1],[[[[4,15,1],[4,15,-1]]],52,48,0.408953,19,1],[[[[9,3,1]],[[9,3,-1]]],4,28,0.417173,27,-1],[[[[7,7,1],[7,7,-1]]],10,34,0.415859,284,-1],[[[[6,29,1],[6,29,-1]]],14,30,0.407719,-47,1],[[[[18,9,1],[18,9,-1]]],18,42,0.413845,-755,-1],[[[[9,7,1]],[[9,7,-1]]],12,32,0.421131,312,-1],[[[[3,5,1],[3,5,-1]]],42,20,0.410372,80,-1],[[[[2,4,1],[2,4,-1]]],24,36,0.413023,9,1]]," +
         "[0.654066,0.478069,0.557677,0.582782,0.575036,0.634584,0.587721,0.648363,0.638677,0.622184,0.617724,0.624051,0.636043,0.611951,0.672508,0.635983,0.669984,0.677474,0.695779,0.689192,0.657189,0.643193,0.683689,0.683984,0.647476,0.660778,0.668047,0.643698,0.648478,0.666227,0.67748,0.702377,0.657186,0.7036,0.675291,0.719929,0.723673,0.694012,0.669077,0.684611,0.681522,0.689258,0.668626,0.691585,0.691914,0.715775,0.711916,0.688387,0.706032,0.727508,0.695985,0.703643]," +
         "9.8523]," +
         "[[[[[[23,11,1]],[[23,11,-1]]],14,40,0.270148,-7267,1],[[[[7,12,1],[7,12,-1]]],46,40,0.23871,-2093,1],[[[[9,15,1],[9,15,-1]]],0,42,0.311489,2757,-1],[[[[19,6,1]],[[19,6,-1]]],0,40,0.342315,2548,-1],[[[[37,7,1]],[[37,7,-1]]],6,14,0.36562,2126,-1],[[[[6,15,1],[6,15,-1]]],22,20,0.371749,-94,1],[[[[9,6,1]],[[9,6,-1]]],40,42,0.387955,-1582,1],[[[[3,9,1],[3,9,-1]]],24,44,0.393157,-57,-1],[[[[6,19,1],[6,19,-1]]],46,24,0.363668,1501,-1],[[[[11,19,1],[11,19,-1]]],4,40,0.381907,589,-1],[[[[6,4,1],[6,4,-1],[6,4,1]]],10,46,0.379702,291,1],[[[[2,6,1],[2,6,-1]]],34,36,0.382581,-17,-1],[[[[19,2,1]],[[19,2,-1]]],22,48,0.39311,196,-1],[[[[4,7,1],[4,7,-1]]],46,54,0.38491,-105,-1],[[[[2,5,1],[2,5,-1]]],44,20,0.394218,-40,1],[[[[9,18,1]],[[9,18,-1]]],14,14,0.400947,2349,-1],[[[[5,3,1]],[[5,3,-1]]],28,18,0.398173,-15,1],[[[[2,9,1],[2,9,-1]]],6,42,0.40701,31,1],[[[[23,3,1]],[[23,3,-1]]],18,14,0.393511,-1251,1],[[[[3,4,1],[3,4,-1]]],32,38,0.389772,24,1],[[[[4,18,1]],[[4,18,-1]]],4,2,0.409087,-1432,1],[[[[11,12,1],[11,12,-1]]],24,34,0.402279,-28,-1],[[[[2,15,1],[2,15,-1]]],34,44,0.414877,22,1],[[[[37,3,1]],[[37,3,-1]]],10,14,0.391444,1805,-1],[[[[3,19,1],[3,19,-1]]],56,40,0.398861,155,1],[[[[7,3,1]],[[7,3,-1]]],34,44,0.39998,-438,1]]," +
         "[0.370141,0.31356,0.45241,0.520485,0.576342,0.591721,0.633866,0.647873,0.571508,0.617879,0.612128,0.619646,0.647744,0.625779,0.650759,0.669303,0.661607,0.68637,0.648834,0.638732,0.692298,0.673021,0.709044,0.643233,0.663509,0.666611]," +
         "5.6432]," +
         "[[[[[[3,5,1],[3,5,-1]]],26,44,0.257129,53,1],[[[[7,12,1],[7,12,-1]]],4,40,0.289519,1015,-1],[[[[4,7,1]],[[4,7,-1]]],46,44,0.352717,-862,1],[[[[5,11,1]],[[5,11,-1]]],36,8,0.366595,844,-1],[[[[6,7,1]],[[6,7,-1]]],6,40,0.38832,78,-1],[[[[15,7,1]],[[15,7,-1]]],24,30,0.392096,1333,-1],[[[[9,3,1]],[[9,3,-1]]],28,22,0.396877,62,-1],[[[[4,11,1]],[[4,11,-1]]],56,40,0.399422,-451,-1],[[[[6,4,1]],[[6,4,-1]]],16,26,0.397423,79,-1],[[[[7,9,1]],[[7,9,-1]]],44,8,0.39911,-420,1],[[[[3,7,1],[3,7,-1]]],40,50,0.403436,18,1],[[[[2,6,1],[2,6,-1]]],14,18,0.386354,88,-1],[[[[2,12,1],[2,12,-1]]],6,40,0.391471,72,1],[[[[2,12,1],[2,12,-1]]],12,34,0.399608,-20,1],[[[[3,15,1],[3,15,-1]]],4,42,0.400528,-80,-1],[[[[11,12,1],[11,12,-1],[11,12,1]]],16,30,0.395247,3978,1],[[[[5,4,1]],[[5,4,-1]]],48,38,0.403161,20,-1],[[[[3,15,1],[3,15,-1]]],44,38,0.413175,40,-1],[[[[11,15,1],[11,15,-1]]],0,6,0.407168,1975,1],[[[[2,4,1],[2,4,-1]]],16,20,0.406053,25,-1],[[[[2,7,1],[2,7,-1]]],26,44,0.402682,-11,-1],[[[[3,6,1],[3,6,-1]]],14,20,0.393289,-51,1],[[[[2,19,1],[2,19,-1]]],24,16,0.404665,-4,1],[[[[19,9,1]],[[19,9,-1]]],2,2,0.40845,-59,1],[[[[3,5,1],[3,5,-1]]],48,26,0.400465,154,-1],[[[[14,19,1],[14,19,-1]]],32,38,0.393894,-1442,1],[[[[15,3,1]],[[15,3,-1]]],24,46,0.407853,-844,1],[[[[2,37,1],[2,37,-1]]],34,22,0.399675,-36,-1],[[[[23,2,1]],[[23,2,-1]]],14,46,0.401985,102,-1],[[[[7,4,1],[7,4,-1]]],48,60,0.399567,-150,-1],[[[[9,2,1]],[[9,2,-1]]],34,14,0.400277,-123,1],[[[[5,23,1]],[[5,23,-1]]],24,12,0.402466,3296,1],[[[[37,2,1]],[[37,2,-1]]],12,26,0.411462,141,-1],[[[[2,12,1],[2,12,-1]]],0,48,0.401303,-13,-1],[[[[2,4,1],[2,4,-1],[2,4,1]]],14,44,0.407886,81,1],[[[[14,6,1],[14,6,-1]]],36,4,0.400775,-1063,-1],[[[[29,9,1]],[[29,9,-1]],[[29,9,1]]],18,10,0.411776,43579,-1],[[[[6,9,1],[6,9,-1]]],28,18,0.387932,-443,-1],[[[[15,9,1]],[[15,9,-1]]],28,34,0.402189,-4703,1],[[[[3,5,1],[3,5,-1]]],34,44,0.392631,28,1],[[[[2,19,1],[2,19,-1]]],48,30,0.405152,30,-1],[[[[5,11,1]],[[5,11,-1]]],44,20,0.406175,250,-1],[[[[6,18,1]],[[6,18,-1]]],28,12,0.412059,4392,1],[[[[3,5,1],[3,5,-1]]],12,22,0.400913,113,-1],[[[[4,4,1]],[[4,4,-1]]],28,10,0.413803,-48,1],[[[[4,6,1]],[[4,6,-1]]],22,28,0.418892,51,-1],[[[[9,4,1]],[[9,4,-1]]],16,38,0.41885,91,-1],[[[[2,4,1],[2,4,-1]]],46,22,0.416862,-22,1],[[[[7,7,1]],[[7,7,-1]]],56,0,0.404885,398,1],[[[[3,6,1],[3,6,-1]]],44,22,0.407688,163,-1],[[[[3,15,1],[3,15,-1]]],22,48,0.405453,-57,-1],[[[[5,3,1]],[[5,3,-1]]],32,28,0.407812,-232,1],[[[[2,4,1],[2,4,-1]]],30,36,0.396779,12,1],[[[[9,2,1]],[[9,2,-1]]],28,28,0.406796,46,-1],[[[[6,3,1]],[[6,3,-1]]],16,42,0.415966,1,-1],[[[[23,2,1]],[[23,2,-1]]],20,44,0.419954,-511,1],[[[[4,5,1],[4,5,-1]]],20,34,0.393481,98,1],[[[[4,15,1],[4,15,-1]]],46,22,0.414846,-718,1]]," +
         "[0.346129,0.407497,0.54492,0.57877,0.634842,0.644997,0.658035,0.665064,0.659538,0.664199,0.676266,0.629605,0.643306,0.665578,0.668135,0.653569,0.675494,0.704086,0.68682,0.683652,0.674151,0.648231,0.679726,0.690475,0.667958,0.649875,0.688771,0.665765,0.672198,0.665464,0.667435,0.673544,0.699126,0.670293,0.688864,0.668822,0.700034,0.633804,0.672768,0.646446,0.681103,0.683997,0.700851,0.669208,0.70591,0.720849,0.720726,0.71486,0.680348,0.688301,0.681953,0.688654,0.657766,0.685761,0.712228,0.724002,0.648752,0.708952]," +
         "11.815]" +
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

   //classifierScales.push_back(0.5);
   //classifierScales.push_back(0.75);
   classifierScales.push_back(1.0);
   classifierScales.push_back(1.1);
   classifierScales.push_back(1.2);
   classifierScales.push_back(1.4);
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
