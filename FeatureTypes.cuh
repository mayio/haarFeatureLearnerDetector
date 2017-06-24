/*
 *
 *  Created on: May 17, 2017
 *      Author: Mario Lüder
 *
 */


#ifndef FEATURETYPES_CUH_
#define FEATURETYPES_CUH_

#include <vector>
#include <string>
#include <assert.h>
#include "stdint.h"

struct FeatureRectangle
{
   FeatureRectangle(const uint32_t w, const uint32_t h, const int32_t t) :
         width(w), height(h), type(t)
   {
   }

   const uint32_t width;
   const uint32_t height;
   const int32_t type;

   FeatureRectangle & operator=(const FeatureRectangle & fr)
   {
      const uint32_t & widthRef = width;
      const uint32_t & heightRef = height;
      const int32_t & typeRef = type;

      const_cast<uint32_t &>(widthRef) = fr.width;
      const_cast<uint32_t &>(heightRef) = fr.height;
      const_cast<int32_t &>(typeRef) = fr.type;
      return *this;
   }

   FeatureRectangle(const FeatureRectangle & fr)
   : width(fr.width)
   , height(fr.height)
   , type(fr.type)
   {
   }
};

class FeatureType
{
public:
   FeatureType(const uint32_t rectWidth, const uint32_t rectHeight) :
         mRect(rectWidth, rectHeight, 0), mFeatureWidth(0), mFeatureHeight(0)
   {
   }

   FeatureType & addRow()
   {
      mFeatureHeight++;
      mFeatureWidth = 0;
      return *this;
   }

   FeatureType & operator <<(int32_t t)
   {
      mTypes.push_back(t);
      mFeatureWidth++;
      return *this;
   }

   void setRect(
         const uint32_t rectWidth,
         const uint32_t rectHeight)
   {
      mRect = FeatureRectangle(rectWidth, rectHeight, 0);
   }

   void setRect(
         const uint32_t rectWidth,
         const uint32_t rectHeight,
         const int32_t  type)
   {
      mRect = FeatureRectangle(rectWidth, rectHeight, type);
   }

   FeatureRectangle mRect;
   uint32_t mFeatureWidth;
   uint32_t mFeatureHeight;
   std::vector<int32_t> mTypes;
};


class FeatureTypes : public std::vector<FeatureType>
{
public:
   FeatureTypes() :
         data(NULL), gpuData(NULL), dataSize(0)
   {
   }
   ;

   FeatureTypes(const FeatureTypes & ft)
   {
      //((std::vector<FeatureType> *)(this))->resize(ft.size());

      for (std::vector<FeatureType>::const_iterator ftIter = ft.begin();
            ftIter != ft.end();
            ++ftIter)
      {
         this->push_back(*ftIter);
      }

      data = NULL;
      gpuData = NULL;
      dataSize = 0;
   }

   FeatureTypes & operator=(const FeatureTypes & ft)
   {
      //((std::vector<FeatureType> *)(this))->resize(ft.size());

      for (std::vector<FeatureType>::const_iterator ftIter = ft.begin();
           ftIter != ft.end();
           ++ftIter)
      {
         this->push_back(*ftIter);
      }

      data = NULL;
      gpuData = NULL;
      dataSize = 0;
      return *this;
   }

   ~FeatureTypes();

   uint32_t getDataSize() const
   {
      return dataSize;
   }
   const uint8_t * getData() const
   {
      return data;
   }
/*
   const uint8_t * getGpuData() const
   {
      return gpuData;
   }
*/
   void copyToConstantMemory();
   static uint8_t * getConstantFeatureData();

private:
   uint8_t * data;
   uint8_t * gpuData;
   uint32_t dataSize;

   /*
    * generate a classifier of consistent memory of the form
    * [
    *    [count feature types, address offset of feature type 1, address offset of feature type 2, ...],
    *    [ // feature type 1
    *       [ feature width, feature height, feature count ],
    *       [ // original feature
    *          [ // rectangle 1
    *             rectangle width, rectangle height, type (black or white)
    *          ],
    *          [ // rectangle 2
    *          ],
    *          ... more rectangle depending on width and height
    *       ],
    *       [ // scaled feature like above
    *          ...
    *       ],
    *       [ // next scaled feature
    *          ...
    *       ]
    *    ],
    *    [ // feature type 2 ...
    *       ...
    *    ],
    * ]
    */
   void generateClassifier(const double scale, const uint32_t windowWidth,
         const uint32_t windowHeight, bool calcOnlySize, uint32_t & memsize);

public:
   struct Scale
   {
      Scale(uint32_t _x, uint32_t _y) :
            x(_x), y(_y)
      {
      }
      uint32_t x, y;
      inline bool operator!=(const Scale & s)
      {
         return !(*this == s);
      }
      inline bool operator==(const Scale & s)
      {
         if (s.x == x && s.y == y)
         {
            return true;
         }

         return false;
      }

      inline Scale & operator=(const Scale & s)
      {
         x = s.x;
         y = s.y;
         return *this;
      }
   };

   void generateClassifier(const double scale, const uint32_t windowWidth,
         const uint32_t windowHeight, bool copyToConst);
};



#endif /* FEATURETYPES_H_ */
