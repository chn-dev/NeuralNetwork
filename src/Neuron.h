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
