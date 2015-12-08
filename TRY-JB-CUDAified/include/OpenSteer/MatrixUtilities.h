#ifndef OPENSTEER_MATRIXUTILITIES_H
#define OPENSTEER_MATRIXUTILITIES_H

// Include std::basic_ostream, std::endl
#include <ostream>


// Include OpenSteer::Matrix
#include "OpenSteer/Matrix.h"

// Include OpenSteer::AbstractLocalSpace
#include "OpenSteer/LocalSpace.h"


namespace OpenSteer {
    
    Matrix localSpaceTransformationMatrix( AbstractLocalSpace const& _localSpace );
    
    void localSpaceTransformationMatrix( Matrix& _transformation, AbstractLocalSpace const& _localSpace );
    
    Matrix translationMatrix( Vec3 const& _translation );
    
    
    // void orthonormalize( Matrix& _matrix );
    
    // Matrix orthonormalize( Matrix const& _matrix );
    
    
    
    template< typename CharT, class Traits >
        std::basic_ostream< CharT, Traits >&
        operator<<( std::basic_ostream< CharT, Traits >& ostr, Matrix const& _matrix ) {
            
            for ( Matrix::size_type i = 0; i < _matrix.rowCount(); ++i ) {
                
                
                
                for ( Matrix::size_type j = 0; j < _matrix.columnCount(); ++j ) {
                    
                    ostr << _matrix( i, j ) << ", ";
                    
                }
                
                ostr << std::endl;
            }
            
            ostr << std::endl;
            
            return ostr;
        }
    
} // namespace OpenSteer


#endif // OPENSTEER_MATRIXUTILITIES_H
