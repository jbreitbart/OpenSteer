#include "OpenSteer/Graphics/GraphicsPrimitivesUtilities.h"

// Include OpenSteer::AbstractVehicle
#include "OpenSteer/AbstractVehicle.h"

// Include OpenSteer::Color
#include "OpenSteer/Color.h"

// Include OpenSteer::sqrtXXX, OpenSteer::square
#include "OpenSteer/Utilities.h"

OpenSteer::SharedPointer< OpenSteer::Graphics::Basic3dSphericalVehicleGraphicsPrimitive > 
OpenSteer::Graphics::makeBasic3dSphericalVehicleGraphicsPrimitive( OpenSteer::AbstractVehicle const& _generate_from,
																   OpenSteer::Color const& _material )
{	
	// Code converted from @c OpenSteer::drawBasic3dSphericalVehicle.
	
	float radius = _generate_from.radius();
	
	// @todo Remove magic number.
	float aspect_x = 0.5f;
	// @todo Remove magic number.
	float aspect_y = sqrtXXX( 1.0f - square( aspect_x ) );
	
	float length_center = radius * _generate_from.forward().length();
	float length = ( 1.0f + aspect_y ) * length_center;
	float width = radius * _generate_from.side().length() * aspect_x * 2.0f;
	float height = radius * _generate_from.up().length() * aspect_x;

	return SharedPointer< Basic3dSphericalVehicleGraphicsPrimitive >( new Basic3dSphericalVehicleGraphicsPrimitive( length_center, length, width, height , _material ) );
}

