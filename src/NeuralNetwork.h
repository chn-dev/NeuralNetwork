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
