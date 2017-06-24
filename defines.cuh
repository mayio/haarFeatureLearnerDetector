/*
 *
 *  Created on: May 17, 2017
 *      Author: Mario Lüder
 *
 */

#ifndef DEFINES_H_
#define DEFINES_H_


#include "stdio.h"

#define DEBUG 1


/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#ifndef CUDA_CHECK_RETURN
#define CUDA_CHECK_RETURN(value) {                                \
   cudaError_t _m_cudaStat = value;                            \
   if (_m_cudaStat != cudaSuccess) {                              \
      fprintf(stderr, "Error %s at line %d in file %s\n",               \
            cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);    \
      exit(1);                                           \
   } }
#endif

#endif
