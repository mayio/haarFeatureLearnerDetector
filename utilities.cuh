/*
 *
 *  Created on: May 17, 2017
 *      Author: Mario Lüder
 *
 */

#ifndef UTILITIES_CUH_
#define UTILITIES_CUH_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <set>
#include <string>
#include <algorithm>

#include "defines.cuh"


static void dumpFreeMemory(const std::string & prefix)
{
   size_t mem_tot_0 = 0;
   size_t mem_free_0 = 0;
   cudaMemGetInfo(&mem_free_0, &mem_tot_0);
   std::cout << prefix << " Free memory:" << mem_free_0 << " Mem total: " << mem_tot_0
         << std::endl;
}

static void deviceSetup()
{
   CUDA_CHECK_RETURN(cudaDeviceReset());

   // Set flag to enable zero copy access
   CUDA_CHECK_RETURN(cudaSetDeviceFlags(cudaDeviceMapHost));

   cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 600);
   dumpFreeMemory("deviceSetup:");
}

static void dumpElapsedTime(const std::string & prefix, cudaEvent_t &start, cudaEvent_t & stop)
{
   float elapsedTime = 0;
   cudaEventElapsedTime(&elapsedTime, start, stop);
   std::cout << prefix << " Elapsed time: "
         << elapsedTime << "ms" << std::endl;
}

// from
// https://stackoverflow.com/questions/8520560/get-a-file-name-from-a-path

static std::string removeExtension( std::string const& filename )
{
    std::string::const_reverse_iterator
                        pivot
            = std::find( filename.rbegin(), filename.rend(), '.' );
    return pivot == filename.rend()
        ? filename
        : std::string( filename.begin(), pivot.base() - 1 );
}

struct MatchPathSeparator
{
    bool operator()( char ch ) const
    {
        return ch == '/';
    }
};

static std::string fileBasename( std::string const& pathname )
{
    std::string baseNameStr =  std::string(
        std::find_if( pathname.rbegin(), pathname.rend(),
                      MatchPathSeparator() ).base(),
        pathname.end() );

    baseNameStr = removeExtension(baseNameStr);
    return baseNameStr;
}

#endif /* HELPER_H_ */
