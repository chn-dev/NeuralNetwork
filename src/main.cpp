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
\file main.cpp
\author Christian Nowak <chnowak@web.de>
\brief The main program.
*/
/*----------------------------------------------------------------------------*/
#include <stdio.h>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "NeuralNetwork.h"
#include "util.h"


/*----------------------------------------------------------------------------*/
/*! 2023-12-11
This function reads a single line from the MNIST CSV file, converts the pixel
data to a vector of 28x28 double values ranging from 0.01 to 1.0 an returns
the marker value.

\see static int readCSV( std::ifstream &infile, std::vector<double> &values )

\param infile The input file stream
\param imin The minimum input value to be expected from the MNIST file
\param imax The maximum input value to be expected from the MNIST file
\param dmin The minimum output value
\param dmax The maximum output value
\param values The vector of doubles which will be filled with the pixel data
\return The marker value
*/
/*----------------------------------------------------------------------------*/
static int readMNIST( std::ifstream &infile, int imin, int imax, double dmin, double dmax, std::vector<double> &values )
{
   std::string line;
   int marker = -1;
   values.clear();

   if( std::getline( infile, line ) )
   {
      line = util::trim( line );
      std::vector<std::string> sVals = util::strsplit( line, ",", false );

      if( sVals.size() > 0 )
      {
         marker = std::stoi( sVals[0] );

         for( int i = 1; i < sVals.size(); i++ )
         {
            int v = std::stoi( sVals[i] );

            double dv = (double)( v - imin ) / (double)( imax - imin );
            dv = ( dv * ( dmax - dmin ) ) + dmin;
            values.push_back( dv );
         }
      }
   }

   return( marker );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-11
In the MNIST CSV file, each line represents a sample of a handwritten digit.
The line begins with the marker (the actual digit as an integer value),
followed by 784 integer values ranging from 0 to 255 representing 28x28 pixels.

This function reads a single line from the MNIST CSV file, converts the pixel
data to a vector of 28x28 double values ranging from 0.01 to 1.0 an returns
the marker value.

\param infile The input file stream
\param values The vector of doubles which will be filled with the pixel data
\return The marker value
*/
/*----------------------------------------------------------------------------*/
static int readMNIST( std::ifstream &infile, std::vector<double> &values )
{
   return( readMNIST( infile, 0, 255, 0.01, 1.0, values ) );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-12
Our neural network has 28x28=784 input neurons for the pixel data of a
handwritten decimal digit and 10 output neurons, each indicating the detection
of a specific decimal digit. I.e. if neuron n (n=0..9) is close to 1.0, that
means that decimal digit n has been detected.
This function converts the value of a decimal digit n, given as an integer
parameter, to an expected output vector consisting of 10 double values where
only element #n is 'trueVal' (usually = 1.0) and all other elements are
'falseVal' (usually = 0.0).

\param digit The digit, ranging from 0 to 9
\param falseVal The double value indicating non-detection
\param trueVal The double value indicating detection
\return The output vector which can be compared to the output vector of our
neural network
*/
/*----------------------------------------------------------------------------*/
std::vector<double> convertToExpectedOut( int digit, double falseVal, double trueVal )
{
   std::vector<double> r;

   for( int i = 0; i < 10; i++ )
   {
      r.push_back( i == digit ? trueVal : falseVal );
   }

   return( r );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
Print usage
*/
/*----------------------------------------------------------------------------*/
void usage( int argc, const char *argv[] )
{
   fprintf( stderr, "Usage: %s mnist_train.csv mnist_test.csv\n", argv[0] );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
Main program
*/
/*----------------------------------------------------------------------------*/
int main( int argc, const char *argv[] )
{
   if( argc < 3 )
   {
      usage( argc, argv );
      return( -1 );
   }

   std::string trainfname = argv[1];
   std::string testfname = argv[2];

   // Initialize the random number generator
   std::srand( std::time( 0 ) );

   // The neuronal network shall have 28x28=784 input neurons,
   // 100 hidden neurons and 10 output neurons (1 for each possible digit 0..9)
   NeuralNetwork nn( { 28 * 28, 100, 10 } );

   // Open the input file
   std::ifstream trainfile = std::ifstream( trainfname );
   if( !trainfile.is_open() )
   {
      fprintf( stderr, "Couldn't open training input file '%s'.\n", trainfname.c_str() );
      return( -1 );
   }

   std::vector<double> inVector;
   int value;

   // *** Train the neural network
   // *** With the first nTrain annotated samples
   printf( "Training..\n" );
   int n;
   for( n = 0;; n++ )
   {
      inVector.clear();
      int digit = readMNIST( trainfile, inVector );
      if( ( digit < 0 ) || ( inVector.size() != 28 * 28 ) )
      {
         if( n < 10 )
         {
            fprintf( stderr, "Error reading MNIST file during training.\nFinished reading %d samples.\n", n );
            trainfile.close();
            return( -1 );
         } else
         {
            break;
         }
      }

      std::vector<double> expectedOutVector = convertToExpectedOut( digit, 0.01, 0.99 );

      // Here's where the training happens
      nn.train( inVector, expectedOutVector, 0.2 );

      // Progress
      if( n % 1000 == 0 )
      {
         printf( "%d..\n", n );
      }
   }

   printf( "Finished training with %d samples.\n", n );

   trainfile.close();

   std::ifstream testfile = std::ifstream( testfname );
   if( !testfile.is_open() )
   {
      fprintf( stderr, "Couldn't open training input file '%s'.\n", testfname.c_str() );
      return( -1 );
   }

   // ** Test the neural network
   // ** With the next 10000 samples
   int nFail = 0;
   int nPass = 0;
   printf( "Testing..\n" );
   for( n = 0;; n++ )
   {
      inVector.clear();
      int digit = readMNIST( testfile, inVector );
      if( ( digit < 0 ) || ( inVector.size() != 28 * 28 ) )
      {
         if( n < 10 )
         {
            fprintf( stderr, "Error reading MNIST file during testing.\nFinished reading %d samples.\n", n );
            testfile.close();
            return( -1 );
         } else
         {
            break;
         }
      }

      // Query the network
      nn.query( inVector );
      std::vector<double> outVector = nn.output();

      // Our network has 10 output neurons, each of which indicating
      // the probability of detection of a specific digit. To determine
      // which digit the network as a whole has detected, we use the
      // number of the output neuron with the highest output value.
      int detectedDigit = util::indexOfMaxValue( outVector );

      // If that detected digit equals the annotated marker of the
      // MNIST dataset, that's a pass
      if( digit == detectedDigit )
      {
         nPass++;
      } else
      {
         nFail++;
      }

      // Progress
      if( n % 1000 == 0 )
      {
         printf( "%d..\n", n );
      }
   }

   testfile.close();

   printf( "Finished testing with %d samples.\n", n );
   printf( "nPass = %d\nnFail = %d\nSuccess rate: %0.1f%%\n",
      nPass, nFail, 100.0 * ( (double)nPass / (double)( nPass + nFail ) ) );
   scanf( "\n" );

   return( 0 );
}
