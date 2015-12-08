#include "OpenSteer/Matrix.h"


// Include std::transform
#include <algorithm>

// Include std::plus
#include <functional>

OpenSteer::Matrix const OpenSteer::identityMatrix( 1.0f, 0.0f, 0.0f, 0.0f,
                                                   0.0f, 1.0f, 0.0f, 0.0f,
                                                   0.0f, 0.0f, 1.0f, 0.0f, 
                                                   0.0f, 0.0f, 0.0f, 1.0f );






OpenSteer::Matrix& 
OpenSteer::operator+=( Matrix& _lhs, Matrix const& _rhs )
{
    // @todo Rewrite for performance after profiling.
    std::transform( _lhs.begin(), _lhs.end(), _rhs.begin(), _lhs.begin(), std::plus< Matrix::value_type >() );
    
    return _lhs;
}

OpenSteer::Matrix 
OpenSteer::operator+( Matrix const& _lhs, Matrix const& _rhs )
{
    Matrix returnValue( _lhs );
    returnValue += _rhs;
    return returnValue;
}








OpenSteer::Matrix 
OpenSteer::operator*( Matrix const& _lhs, Matrix const& _rhs )
{
    Matrix::value_type const initValue = Matrix::value_type();
    Matrix returnValue( initValue );
    
    for ( Matrix::size_type i = 0; i < _lhs.rowCount(); ++i ) {
        
        for ( Matrix::size_type j = 0; j < _rhs.columnCount(); ++j ) {
            
            for ( Matrix::size_type k = 0; k < _lhs.columnCount(); ++k ) {
                
                returnValue( i, j ) += _lhs( i, k ) * _rhs( k, j );
                
            }
            
        }
        
    }
    
    
    return returnValue;
}
