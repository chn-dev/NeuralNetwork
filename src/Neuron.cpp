/*----------------------------------------------------------------------------*/
/*!
\file Neuron.cpp
\author Christian Nowak <chnowak@web.de>
\brief Implementation of the class Neuron.
*/
/*----------------------------------------------------------------------------*/
#include <math.h>

#include "Neuron.h"
#include "util.h"


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
Constructor
\param layer The number of the layer where the neuron is located within the
network
\param numInput The number of inputs of this neuron
*/
/*----------------------------------------------------------------------------*/
Neuron::Neuron( int layer, int nInputs ) :
   m_nLayer( layer ),
   m_numInputs( nInputs ),
   m_Error( 0.0 ),
   m_Output( 0.0 )
{
   m_Weights.resize( m_numInputs );
   m_Inputs.resize( m_numInputs );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
Destructor
*/
/*----------------------------------------------------------------------------*/
Neuron::~Neuron()
{

}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
Set the output error (the difference betweeb the output and the expected output).
\param e The error
*/
/*----------------------------------------------------------------------------*/
void Neuron::setError( double e )
{
   m_Error = e;
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
\return The output error
*/
/*----------------------------------------------------------------------------*/
double Neuron::error() const
{
   return( m_Error );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
\return The output, after the input has been processed with query()
*/
/*----------------------------------------------------------------------------*/
double Neuron::output() const
{
   return( m_Output );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
Feed the neuron with an input vector and calculate the output. The size of the
input vector must be equal to the number of inputs of the neuron.

The output is calculated with:

$$ o = { 1 \over { 1 + e ^ { - \sum_{k=0}^{numInputs-1} { w_k i_k } } } } $$

Whereby

$$ w_k $$

are the input weights and

$$ i_k $$

are the input values.

After the query, the resulting output can be acquired with the output() function.

\return true on success or false on failure
*/
/*----------------------------------------------------------------------------*/
bool Neuron::query( std::vector<double> inputVector )
{
   if( m_Inputs.size() != inputVector.size() )
   {
      return( false );
   }

   for( int i = 0; i < inputVector.size(); i++ )
   {
      m_Inputs[i] = inputVector[i];
   }

   return( query() );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
Convenience function for the special case where the input vector's length is 1.
\return true on success
*/
/*----------------------------------------------------------------------------*/
bool Neuron::query( double v )
{
   std::vector<double> inputVector = std::vector<double>( { v } );
   return( query( inputVector ) );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
Process the internally stored input values and store the result internally.
\return true on success
*/
/*----------------------------------------------------------------------------*/
bool Neuron::query()
{
   double v = 0.0;

   // Sanity check
   if( m_Weights.size() != m_Inputs.size() )
   {
      return( false );
   }

   // Calculate the weighted sum of the inputs
   for( int i = 0; i < m_Inputs.size(); i++ )
   {
      v += m_Inputs[i] * m_Weights[i];
   }

   // If this neuron is part of the input layer within the network, don't
   // apply the activation function.
   if( m_nLayer == 0 )
   {
      m_Output = v;
   } else
   {
      m_Output = 1.0 / ( 1.0 + exp( -v ) );
   }

   return( true );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
Randomize all input weights of the neuron. The input weights will be set to
random values ranging from

$$ \sqrt{ -{ 1 \over numInputs } } $$

to

$$ \sqrt{ +{ 1 \over numInputs } } $$
*/
/*----------------------------------------------------------------------------*/
void Neuron::randomizeWeights()
{
   for( int i = 0; i < m_Weights.size(); i++ )
   {
      if( m_nLayer == 0 )
      {
         m_Weights[i] = 1.0;
      } else
      {
         m_Weights[i] = util::randomValue( -1.0 / sqrt( m_numInputs ), 1.0 / sqrt( m_numInputs ) );
      }
   }
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
This is the implementation of the "learning" process. Call this function after
using query() followed by setError() to adjust the input weights according to

- The current input weights
- The output
- The error
- The learning rate

\param alpha The learning rate
*/
/*----------------------------------------------------------------------------*/
void Neuron::adjustWeights( double alpha )
{
   for( int i = 0; i < m_Weights.size(); i++ )
   {
      // gradient
      double gradient = - m_Error * output() * ( 1.0 - output() ) * m_Inputs[i];

      // Adjust the weight inproportion to the gradient and the learning rate
      m_Weights[i] = m_Weights[i] - ( alpha * gradient );
   }
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
\return The number of inputs of this neuron
*/
/*----------------------------------------------------------------------------*/
int Neuron::numInputs() const
{
   return( m_numInputs );
}


/*----------------------------------------------------------------------------*/
/*! 2023-12-15
Retrieve a specific input weight of this neuron.
\param n The index of the input weight, ranging from 0 to numInputs() - 1
\return The nth input weight
*/
/*----------------------------------------------------------------------------*/
double Neuron::weight( int n ) const
{
   if( n < 0 || n >= m_Weights.size() )
      return( NAN );
   else
      return( m_Weights[n] );
}
