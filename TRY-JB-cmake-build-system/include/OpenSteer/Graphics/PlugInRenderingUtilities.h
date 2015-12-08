/**
 * @file
 *
 * Helper functions for plugins to set the annotation logger or to draw the vehicles using
 * C++ STL algorithms for containers like @c std::for_each.
 */

#ifndef OPENSTEER_PLUGIN_RENDERING_UTILITIES_H
#define OPENSTEER_PLUGIN_RENDERING_UTILITIES_H


// Include std::unary_function
#include <functional>

// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"

// Include OpenSteer::Graphics::RenderFeeder, OpenSteer::Graphics::RenderFeeder::InstanceId
#include "OpenSteer/Graphics/RenderFeeder.h"

// Include OpenSteer::Graphics::AnnotationLogger
#include "OpenSteer/Graphics/GraphicsAnnotationLogger.h"

// Include OpenSteer::localSpaceTransformationMatrix
#include "OpenSteer/MatrixUtilities.h"



namespace OpenSteer {
	
	namespace Graphics {
	
	
	template< class Vehicle >
	class DrawVehicle : public std::unary_function< Vehicle*, void > {
	public:
        DrawVehicle( SharedPointer< Graphics::RenderFeeder > const& _renderFeeder,
                     Graphics::RenderFeeder::InstanceId _vehicleId )
        : renderFeeder_( _renderFeeder ), vehicleId_( _vehicleId ) {
			
        }
		
        void operator() ( Vehicle const* _vehicle ) {
            renderFeeder_->render( localSpaceTransformationMatrix( *_vehicle ),
                                   vehicleId_ );
        }
		
	private:
        SharedPointer< Graphics::RenderFeeder > renderFeeder_;
        Graphics::RenderFeeder::InstanceId vehicleId_;
    }; // class DrawVehicle
	
	
	
	
	template< class Vehicle >
    class SetAnnotationLogger : public std::unary_function< Vehicle*, void > {
	public:
        SetAnnotationLogger( SharedPointer< Graphics::GraphicsAnnotationLogger > _annotationLogger ) : annotationLogger_( _annotationLogger ) {
        }
		
        void operator()( Vehicle* _vehicle ) {
            _vehicle->setAnnotationLogger( annotationLogger_ );
        }
		
		
	private:
        SharedPointer< Graphics::GraphicsAnnotationLogger > annotationLogger_;
		
    }; // class SetAnnotationLogger
	
	
	
	template< class Vehicle >
    class SetRenderFeeder : public std::unary_function< Vehicle*, void > {
	public:
        SetRenderFeeder( SharedPointer< Graphics::RenderFeeder > _renderFeeder ) : renderFeeder_( _renderFeeder ) {
        }
		
        void operator()( Vehicle* _vehicle ) {
            _vehicle->setRenderFeeder( renderFeeder_ );
        }
		
		
	private:
        SharedPointer< Graphics::RenderFeeder > renderFeeder_;
		
    }; // class SetRenderFeeder
	
	
	
} // namespace Graphics
	
} // namespace OpenSteer



#endif // OPENSTEER_PLUGIN_RENDERING_UTILITIES_H
