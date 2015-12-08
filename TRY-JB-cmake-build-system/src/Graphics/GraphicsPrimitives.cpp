#include "OpenSteer/Graphics/GraphicsPrimitives.h"

// Include std::swap
#include <algorithm>


OpenSteer::Graphics::GraphicsPrimitive::~GraphicsPrimitive()
{
    // Nothing to do.
}




void 
OpenSteer::Graphics::LineGraphicsPrimitive::swap( LineGraphicsPrimitive& other )
{
    OpenSteer::swap( material_, other.material_ );
    OpenSteer::swap( lineBegin_, other.lineBegin_ );
    OpenSteer::swap( lineEnd_, other.lineEnd_ );
}



void
OpenSteer::Graphics::Vehicle2dGraphicsPrimitive::swap( Vehicle2dGraphicsPrimitive& other )
{
    OpenSteer::swap( material_, other.material_ );
    std::swap( radius_, other.radius_ );
}


void
OpenSteer::Graphics::Basic3dSphericalVehicleGraphicsPrimitive::swap( Basic3dSphericalVehicleGraphicsPrimitive& other )
{
    OpenSteer::swap( material_, other.material_ );
    std::swap( material_variation_factor_, other.material_variation_factor_ );
	std::swap( length_center_, other.length_center_ );
	std::swap( length_, other.length_ );
	std::swap( width_, other.width_ );
	std::swap( height_, other.height_ );
}



void
OpenSteer::Graphics::CircleGraphicsPrimitive::swap( CircleGraphicsPrimitive& other )
{
    OpenSteer::swap( material_, other.material_ );
    std::swap( radius_, other.radius_ );
    std::swap( segmentCount_, other.segmentCount_ );
}



void
OpenSteer::Graphics::DiscGraphicsPrimitive::swap( DiscGraphicsPrimitive& other )
{
    OpenSteer::swap( material_, other.material_ );
    std::swap( radius_, other.radius_ );
    std::swap( segmentCount_, other.segmentCount_ );
}


void 
OpenSteer::Graphics::FloorGraphicsPrimitive::swap( FloorGraphicsPrimitive& _other )
{
    std::swap( breadth_, _other.breadth_ );
    std::swap( length_, _other.length_ );
    material_.swap( _other.material_ );
}


void
OpenSteer::Graphics::TextAt2dLocationGraphicsPrimitive::swap( TextAt2dLocationGraphicsPrimitive& other )
{
    OpenSteer::swap( material_, other.material_ );
    OpenSteer::swap( position_, other.position_ );
    std::swap( relativePosition_, other.relativePosition_ );
    text_.swap( other.text_ );
}


void
OpenSteer::Graphics::TextAt3dLocationGraphicsPrimitive::swap( TextAt3dLocationGraphicsPrimitive& other )
{
    OpenSteer::swap( material_, other.material_ );
    text_.swap( other.text_ );
}

