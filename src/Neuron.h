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
\file Neuron.h
\author Christian Nowak <chnowak@web.de>
\brief Headerfile for class Neuron
*/
/*----------------------------------------------------------------------------*/
#ifndef __NEURON_H__
#define __NEURON_H__

#include <vector>

/*----------------------------------------------------------------------------*/
/*!
\class Neuron
\date  2023-12-12
*/
/*----------------------------------------------------------------------------*/
class Neuron
{
public:
   Neuron( int layer, int nInputs );
   ~Neuron();

   void setError( double e );
   void adjustWeights( double alpha );
   double error() const;
   double weight( int n ) const;
   int numInputs() const;
   double output() const;
   bool query( double v );
   bool query( std::vector<double> inputVector );
   void randomizeWeights();

private:
   bool query();

private:
   std::vector<double> m_Weights;
   std::vector<double> m_Inputs;

   double m_Error;
   double m_Output;

   int m_numInputs;
   int m_nLayer;
};

#endif
