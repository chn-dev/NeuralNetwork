/*----------------------------------------------------------------------------*/
/*!
\file util.h
\author Christian Nowak <chnowak@web.de>
\brief Headerfile for some utility functions
*/
/*----------------------------------------------------------------------------*/
#ifndef __UTIL_H__
#define __UTIL_H__

#include <vector>
#include <string>

namespace util
{
   int indexOfMaxValue( const std::vector<double> &a );
   std::string trim( std::string s );
   std::vector<std::string> strsplit( std::string str, std::string sep, bool keepEmpty );
   double randomValue( double min, double max );
}

#endif
