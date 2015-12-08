#include "OpenSteer/MatrixUtilities.h"

OpenSteer::Matrix 
OpenSteer::localSpaceTransformationMatrix( OpenSteer::AbstractLocalSpace const& _localSpace )
{
    return Matrix( _localSpace.side(), 
                   _localSpace.up(), 
                   _localSpace.forward(), 
                   _localSpace.position() ); 
}



void 
OpenSteer::localSpaceTransformationMatrix( OpenSteer::Matrix& _transformation, 
                                        OpenSteer::AbstractLocalSpace const& _localSpace )
{
    _transformation.assign( _localSpace.side(), 
                            _localSpace.up(), 
                            _localSpace.forward(), 
                            _localSpace.position() );
}


OpenSteer::Matrix 
OpenSteer::translationMatrix( Vec3 const& _translation ) 
{
    Matrix result( identityMatrix );
    
    result.assignColumn( 3, _translation[ 0 ], _translation[ 1 ], _translation[ 2 ],  1.0f );
    
    return result;
}



// void orthonormalize( Matrix& _matrix );

// Matrix orthonormalize( Matrix const& _matrix );

