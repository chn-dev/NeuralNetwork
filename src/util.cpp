/*******************************************************************************
 *  Copyright (c) 2024 Christian Nowak <chnowak@web.de>                        *
 *   This file is part of NeuralNetwork.                                       *
 *                                                                             *
 *  NeuralNetwork is free software: you can redistribute it and/or modify it   *
 *  under the terms of the GNU General Public License as published by the Free *
 *  Software Foundation, either version 3 of the License, or (at your option)  *
 *  any later version.                                                         *
 *                                                                             *          
 *  NeuralNetwork is distributed in the hope that it will be useful, but       * 
 *  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY *
 *  or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License    *
 *  for more details.                                                          *
 *                                                                             *
 *  You should have received a copy of the GNU General Public License along    *
 *  with NeuralNetwork. If not, see <https://www.gnu.org/licenses/>.           *
 *******************************************************************************/


/*----------------------------------------------------------------------------*/
/*!
\file util.cpp
\author Christian Nowak <chnowak@web.de>
\brief Some utility functions
*/
/*----------------------------------------------------------------------------*/

#include "util.h"

namespace util
{
   /*----------------------------------------------------------------------------*/
   /*! 2023-12-15
   \param a A vector of doubles
   \return The index of the double with the highest value within the vector
   */
   /*----------------------------------------------------------------------------*/
   int indexOfMaxValue( const std::vector<double> &a )
   {
      if( a.size() < 1 )
      {
         return( -1 );
      }

      int r = 0;
      double v = a[0];

      for( int i = 0; i < a.size(); i++ )
      {
         if( a[i] > v )
         {
            v = a[i];
            r = i;
         }
      }

      return( r );
   }


   /*----------------------------------------------------------------------------*/
   /*! 2023-12-15
   \param s The string to be trimmed
   \return The string with whitespaces trimmed from both sides
   */
   /*----------------------------------------------------------------------------*/
   std::string trim( std::string s )
   {
      std::string ws = "\r\n\t ";
      size_t pos = s.find_first_not_of( ws );
      if( pos != std::string::npos )
      {
         s = s.substr( pos );
      }

      pos = s.find_first_of( ws );
      if( pos != std::string::npos )
      {
         s = s.substr( 0, pos );
      }

      return( s );
   }


   /*----------------------------------------------------------------------------*/
   /*! 2023-12-15
   Split a string into substrings where a given separator occurs.
   \param str The string to split
   \param sep The separator
   \param keepEmpty If false, empty substrings will be omitted
   \return The vector of substrings
   */
   /*----------------------------------------------------------------------------*/
   std::vector<std::string> strsplit( std::string str, std::string sep, bool keepEmpty )
   {
      std::vector<std::string> r;

      size_t pos;
      size_t curPos = 0;
      while( ( pos = str.find( sep, curPos ) ) != std::string::npos )
      {
         std::string s = str.substr( curPos, pos - curPos );
         if( keepEmpty || s.size() > 0 )
         {
            r.push_back( s );
         }
         curPos = pos + sep.size();
      }

      if( curPos <= str.size() )
      {
         std::string s = str.substr( curPos );
         if( keepEmpty || s.size() > 0 )
         {
            r.push_back( s );
         }
      }
      return( r );
   }


   /*----------------------------------------------------------------------------*/
   /*! 2023-12-15
   Generate a random floating point value within given limits.
   \param min The lower limit
   \param max The upper limit
   \return The generated random value
   */
   /*----------------------------------------------------------------------------*/
   double randomValue( double min, double max )
   {
      if( max < min )
      {
         double tmp = min;
         min = max;
         max = min;
      }

      double v = (double)std::rand() / (double)RAND_MAX; // v = [0.0 .. 1.0]

      return( ( v * ( max - min ) ) + min );
   }
}
