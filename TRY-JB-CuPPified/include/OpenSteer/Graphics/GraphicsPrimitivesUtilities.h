#ifndef OPENSTEER_GRAPHICS_PRIMITIVES_UTILITIES_H
#define OPENSTEER_GRAPHICS_PRIMITIVES_UTILITIES_H

// Include opensteer::graphics::Basic3dSphericalVehicleGraphicsPrimitive
#include "OpenSteer/Graphics/GraphicsPrimitives.h"

// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"


namespace OpenSteer {
	
	
	// Forward declaration
	class AbstractVehicle;
	
namespace Graphics {
	
	SharedPointer< Basic3dSphericalVehicleGraphicsPrimitive > 
	makeBasic3dSphericalVehicleGraphicsPrimitive( AbstractVehicle const& _generate_from,
												  Color const& _material );
	
	
	
} // namespace Graphics
	
	
} // namespace OpenSteer


#endif // OPENSTEER_GRAPHICS_PRIMITIVES_UTILITIES_H
