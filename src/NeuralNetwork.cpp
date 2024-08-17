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
\file NeuralNetwork.cpp
\author Christian Nowak <chnowak@web.de>
\brief Implementation of the class NeuralNetwork with an arbitrary number of
layers and an arbitrary number of neurons in each layer.
*/
/*----------------------------------------------------------------------------*/
#include "NeuralNetwork.h"

/*----------------------------------------------------------------------------*/
/*! 2023-12-12
Constructor
\param numNeurons A vector of integers indicating the number of desired neurons
in each layer, from left (input layer) to right (output layer).
*/
/*----------------------------------------------------------------------------*/
NeuralNetwork::NeuralNetwork( std::vector<int> numNeurons )
{
   for( int i = 0; i < numNeurons.size(); i++ )
   {
      std::vector<Neuron> neurons;
      for( int j = 0; j < numNeurons[i]; j++ )
      {
         Neuron neuron = Neuron( i, i == 0 ? 1 : m_Network[i - 1].size() );
         neurons.push_back( neuron );
      }
      m_Network.push_back( neurons );
   }

   randomizeWeights();
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-12
Destructor
*/
/*----------------------------------------------------------------------------*/
NeuralNetwork::~NeuralNetwork()
{
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
\return The number of layers of this network
*/
/*----------------------------------------------------------------------------*/
int NeuralNetwork::numLayers() const
{
   return( m_Network.size() );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-12
Adjust the input weights of all neurons in proportion to the error. The error
is the difference between the network response to an input vector and the
expected response.

\param input The input vector
\param expectedResult The expected response of the network
\param alpha The learning rate, ranging from 0.0 to 1.0
*/
/*----------------------------------------------------------------------------*/
void NeuralNetwork::train( std::vector<double> input, std::vector<double> expectedResult, double alpha )
{
   // **** 1st step: Query the network with the training sample
   query( input );
   std::vector<double> result = output();

   // If the sizes of the expected result and the network's response
   // are not the same, we can't train.
   if( result.size() != expectedResult.size() )
   {
      return;
   }

   // **** 2nd step: Determine the error of the network
   // The error is the difference between the network response
   // and the expected output.
   std::vector<double> err;
   for( int i = 0; i < result.size(); i++ )
   {
      err.push_back( expectedResult[i] - result[i] );
   }

   // Set the output error values in the last layer
   for( int i = 0; i < err.size(); i++ )
   {
      m_Network[numLayers() - 1][i].setError( err[i] );
   }

   // **** 3rd step: Successively backpropagate the error
   // from the last to the second layer.
   // The first (input) layer does not have an error.
   for( int i = numLayers() - 1; i >= 2; i-- )
   {
      // Propagate the error from layer i to layer i - 1
      backPropagateError( i );
   }

   // **** 4th step: Successively adjust the input weights
   // of all layers in proportion to their errors.
   for( int i = numLayers() - 1; i >= 1 ; i-- )
   {
      for( int j = 0; j < m_Network[i].size(); j++ )
      {
         m_Network[i][j].adjustWeights( alpha );
      }
   }
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-14
Backpropagate the error vector of a specific layer to the previous layer.

The error of neuron n of layer nLayer is:

$$ e_{n,nLayer} $$

The input weight i of neuron n of layer nLayer is:

$$ w_{i,n,nLayer} $$

The error of neuron n of layer nLayer - 1 (the 'previous' layer) is then the
weighted sum

$$ e_{n,nLayer - 1} = \sum_{k=0}^{numLayers - 1} e_{k,nLayer} * w_{n,k,nLayer} $$

\param nLayer The index of the layer to backpropagate to the previous layer

*/
/*----------------------------------------------------------------------------*/
void NeuralNetwork::backPropagateError( int nLayer )
{
   // Sanity check
   if( nLayer >= numLayers() || nLayer < 2 )
      return;
   int prevLayer = nLayer - 1;

   for( int i = 0; i < m_Network[prevLayer].size(); i++ )
   {
      double ei = 0.0;

      for( int j = 0; j < m_Network[nLayer].size(); j++ )
      {
         ei += m_Network[nLayer][j].error() * m_Network[nLayer][j].weight( i );
      }

      m_Network[prevLayer][i].setError( ei );
   }
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-14
\return The output vector of the last layer
*/
/*----------------------------------------------------------------------------*/
std::vector<double> NeuralNetwork::output()
{
   return( output( numLayers() - 1 ) );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-14
Returns the output vector of a specific layer.
\param nLayer the index of the layer
\return The output vector of the specified layer
*/
/*----------------------------------------------------------------------------*/
std::vector<double> NeuralNetwork::output( int nLayer )
{
   std::vector<double> r;

   // Sanity check
   if( nLayer >= numLayers() || nLayer < 0 )
   {
      return( r );
   }

   for( int i = 0; i < m_Network[nLayer].size(); i++ )
   {
      r.push_back( m_Network[nLayer][i].output() );
   }

   return( r );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-14
Feed an input vector into the input layer of the network and calculate the
resulting ouptput of the output layer according to the input weights of all
neurons.
After invoking the query() method, the output() method can be used to collect
the result from the output layer of the network.

\param inputVector The input vector. Its length must be equal to the number of
input neurons.
\return true on success, false on failure
*/
/*----------------------------------------------------------------------------*/
bool NeuralNetwork::query( std::vector<double> inputVector )
{
   // Sanity checks
   if( m_Network.size() < 1 )
   {
      return( false );
   }

   if( ( m_Network[0].size() != inputVector.size() ) ||
       ( m_Network[0].size() < 1 ) )
   {
      return( false );
   }

   // Query the first layer.
   // By definition, each neuron of the first layer has only one input,
   // without any weights.
   for( int j = 0; j < inputVector.size(); j++ )
   {
      if( !m_Network[0][j].query( inputVector[j] ) )
         return( false );
   }

   for( int i = 1; i < m_Network.size(); i++ )
   {
      // Collect input vector from previous layer
      std::vector<double> input = output( i - 1 );

      // Feed that into this layer
      for( int j = 0; j < m_Network[i].size(); j++ )
      {
         m_Network[i][j].query( input );
      }
   }

   return( true );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
Randomize all input weights of all neurons. The input weights will be set to
random values ranging from

$$ \sqrt{ -{ 1 \over numInputs } } $$

to

$$ \sqrt{ +{ 1 \over numInputs } } $$
*/
/*----------------------------------------------------------------------------*/
void NeuralNetwork::randomizeWeights()
{
   for( int i = 0; i < m_Network.size(); i++ )
   {
      for( int j = 0; j < m_Network[i].size(); j++ )
      {
         m_Network[i][j].randomizeWeights();
      }
   }
}
