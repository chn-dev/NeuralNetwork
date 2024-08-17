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
\file NeuralNetwork.h
\author Christian Nowak <chnowak@web.de>
\brief Headerfile for class NeuralNetwork
*/
/*----------------------------------------------------------------------------*/
#ifndef __NEURALNETWORK_H__
#define __NEURALNETWORK_H__

#include <vector>

#include "Neuron.h"

/*----------------------------------------------------------------------------*/
/*!
\class NeuralNetwork
\date  2023-12-12
*/
/*----------------------------------------------------------------------------*/
class NeuralNetwork
{
public:
   NeuralNetwork( std::vector<int> numNeurons );
   ~NeuralNetwork();

   void train( std::vector<double> input, std::vector<double> expectedResult, double alpha );
   bool query( std::vector<double> inputVector );

   std::vector<double> output();
   void randomizeWeights();
   int numLayers() const;

private:
   std::vector<double> output( int nLayer );
   void backPropagateError( int nLayer );

private:
   std::vector<std::vector<Neuron> > m_Network;
};

#endif
