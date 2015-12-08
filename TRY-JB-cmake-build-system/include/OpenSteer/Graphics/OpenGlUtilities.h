#ifndef OPENSTEER_GRAPHICS_OPENGLUTILITIES_H
#define OPENSTEER_GRAPHICS_OPENGLUTILITIES_H


/**
 * @todo Don't create the return pointer in every function but pass it in as an 
 *       argument and fill it. This allows reuse of render meshes.
 */

// Include std::copy, std::fill_n, std::generate_n
#include <algorithm>


// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"

// Include OpenSteer::Color
#include "OpenSteer/Color.h"

// Include OpenSteer::Graphics::OpenGlImage
#include "OpenSteer/Graphics/OpenGlImage.h"

// Include OpenSteer::Graphics::OpenGlRenderMesh
#include "OpenSteer/Graphics/OpenGlRenderMesh.h"

// Include OpenSteer::Trail
#include "OpenSteer/Trail.h"

// Include OpenSteer::size_t
#include "OpenSteer/StandardTypes.h"

// Include OpenSteer::Vec3
#include "OpenSteer/Vec3.h"


namespace {
    
    /**
     * Based on an idea from Ray Lischner, C++ in a Nutshell, O'Reilly 2003, 
     * p. 339.
     */
    template < typename T > class Series {
    public:
        Series( T const& _initialValue ) : next_( _initialValue ) {}
        T operator()() { return next_++; }
        
    private:
        T next_;
    }; // class Series
    
    
    class InsertVecElements {
    public:
        InsertVecElements( OpenSteer::Vec3 const& _vec ) : vec_( _vec ), counter_( 0 ) {}
        
        float operator()() { return vec_[ counter_++ % OpenSteer::Vec3::size() ]; }
        
    private:
        OpenSteer::Vec3 const vec_;
        OpenSteer::size_t counter_; 
        
    }; // class InsertVecElements
    
    
    
}




namespace OpenSteer {
    
    
    namespace Graphics {

        /**
         * Creates an image of a checkerboard that can be used as a texture 
         * image.
         *
         * @attention Because of compatibility issues with older OpenGL 
         *            libraries @a _width and @a _height should be powers of 
         *            @c 2.
         *
         * @param _subdivisions Hint how many checks along one axis should be 
         *        generated. Mustn't be @c 0.
         */
        SharedPointer< OpenGlImage > makeCheckerboardImage( size_t _width, size_t _height, size_t _subdivisions, Color const& _color0, Color const& _color1 );
        
        
        /**
         * Createss an @c OpenGlRenderMesh that represents a line between 
         * @a lineBegin and @a lineEnd and has the color @a color.
         */
        SharedPointer< OpenGlRenderMesh > createLine( Vec3 const& lineBegin, Vec3 const& lineEnd, Color const& color );
        
        /**
         * Creates an @c OpenGlRenderMesh that represents a vehicle triangle 
         * that fits into a circle of radius @a radius and has the color 
         * @a color.
         */
        SharedPointer< OpenGlRenderMesh > createVehicleTriangle( float radius, Color const& color ) ;
        
		
		/**
		 * Creates an @c OpenGlRenderMesh that represents a basic 3d spherical vehicle. It looks
		 * like a triangular arrowhead. Its center of mass is at 
		 * @a length_center when looking from the nose to the back.
		 *
		 * @param material_variation_factor varies @a color for the different vertices of the 
		 *                                  vehicle.
		 */
		SharedPointer< OpenGlRenderMesh > createBasic3dSphericalVehicle( float length_center,
																		 float length,
																		 float width,
																		 float height, 
																		 Color const& color,
																		 float material_variation_factor );
		
		
		
		
        /**
         * Creates an @c OpenGlRenderMesh representing a circle of @a radius
         * with @a segmentCount segments to approximate an ideal circle and with
         * color @a color.
         */
        SharedPointer< OpenGlRenderMesh > createCircle( float radius, size_t segmentCount, Color const& color );
        
        /**
         * Creates an @c OpenGlRenderMesh representing a disc of radius 
         * @a radius, with @a segmentCount segments to approximate an ideal 
         * disc, and with color @a color. A triangle fan is created therefore.
         */
        SharedPointer< OpenGlRenderMesh > createDisc( float radius, size_t segmentCount, Color const& color );
        
        /**
         * Creates an @c OpenGlRenderMesh representing a floor with a texture
         * identified by @a _textureId that has already been added to the
         * renderer.
         */
        SharedPointer< OpenGlRenderMesh > createFloor( float _broadth, 
                                                       float _length, 
                                                       Color const& _color, 
                                                       OpenGlRenderMesh::TextureId _textureId,
                                                       float _broadthTextureScale = 10.0f,
                                                       float _lengthTextureScale = 10.0f );
        
        
        
        
        /**
         * Creates an @c OpenGlRenderMesh representing a trail @a _trail by a 
         * collection of lines. The main color is @a _mainColor and every 
         * @a _tickDuration time ticks a line is drawn in @a _tickColor.
         *
         * The trail steps lines get more transparent the older they are with a
         * minimum opacity of @a _minOpacityValue.
         */
        template< size_t LineCount >
            SharedPointer< OpenGlRenderMesh > createTrail( Trail< LineCount > const& _trail, 
                                                           Color const& _mainColor, 
                                                           Color const& _tickColor, 
                                                           float _tickDuration = 1.0f, 
                                                           float _minOpacityValue = 0.05f ) {
                size_t const trailPositionCount = _trail.positionCount();
                SharedPointer< OpenGlRenderMesh > primitive( new OpenGlRenderMesh( OpenGlRenderMesh::LINES, trailPositionCount ) );
                
                
                if ( 1 >= trailPositionCount ) {
                    return primitive;
                }
                
                
                // @todo Put every @c insertAtBack into its own loop to enable
                //       easy low-level parallelization if needed. Profile first!
                // Set the vertices position and normal data and set the
                // associated indices.
                //for ( size_t i = 0; i != trailPositionCount; ++i ) {
                //   insertAtBack( primitive->vertices_, _trail.footstepPosition( i ) );
                    // Colors are inserted below.
                    //insertAtBack( primitive->normals_, Vec3::up );
                    //primitive->indices_.push_back( i );
                //}
                
                // @todo Remove if by changing the way to get data from a trail.
                // if ( 0 < _trail.positionCount() ) {
                    primitive->vertices_.resize( trailPositionCount * Vec3::size() );
                    std::copy( _trail.footstepPosition( 0 ).data(), _trail.footstepPosition( 0 ).data() + ( trailPositionCount * Vec3::size() ), primitive->vertices_.begin() );
                // }
                // std::fill_n( primitive.normals_.begin(), trailPositionCount, Vec3::up );
                
                
                // for ( size_t i = 0; i != trailPositionCount; ++i ) {
                //    insertAtBack( primitive->normals_, Vec3::up ); 
                //}
                
                primitive->normals_.resize( trailPositionCount * Vec3::size() );
                std::generate_n( primitive->normals_.begin(), trailPositionCount * Vec3::size(), InsertVecElements( Vec3::up ) );
                
                
                
                // for ( size_t i = 0; i != trailPositionCount; ++i ) {
                //     primitive.indices_.push_back( i ); 
                // }
                
                primitive->indices_.resize( trailPositionCount );
                std::generate_n( primitive->indices_.begin(), trailPositionCount, Series< size_t >( 0 ) );
                
                
                
                // Set the vertex colors.
                // Idea: every footstep position has a certain time distance to 
                // the last position of the trail. Map this distance to the 
                // values between @c 1.0f and @c minOpacityValue.
                
                
                float const maxFractionFactor = 1.0f - _minOpacityValue;
                // float duration = _trail.duration();
                
                
                // if ( 0 != trailPositionCount ) {
                    float duration = _trail.lastFootstepPositionTime() - _trail.footstepPositionTimeAtTime( 0 );
                    if ( duration <= 0.0f) {
                        duration = _trail.duration();
                    }
                // }
                
                float const reciprocalDuration = 1.0f / duration;
                
                
                size_t const trailLineCount = _trail.footstepCount();
                 /*
                float reciprocalTrailLineCount = 1.0f;
                if ( 0 != trailLineCount ) {
                    reciprocalTrailLineCount = 1.0f / static_cast< float >( trailLineCount );
                }
                */
                // primitive->colors_.resize( trailPositionCount * 2 * 4 );
                
                for ( size_t i = 0; i != trailLineCount; ++i ) {
                    
                    Color drawColor( _trail.footstepAtTick( i, _tickDuration ) ? _tickColor : _mainColor );
                    
                    
                    /*
                    // Look at the two positions belonging to each footstep.
                    for ( size_t j = 0; j != 2; ++j ) {
                        
                        // No explicit handling of negative values because OpenGL will handle
                        // it gracefully though perhaps in an ugly way.
                        float const fraction = ( _trail.lastFootstepPositionTime() - _trail.footstepPositionTime( i * 2 + j ) ) * reciprocalDuration;
                        float const opacity = 1.0f - fraction * maxFractionFactor;
                        
                        drawColor.setA( opacity );
                        insertAtBack( primitive->colors_, drawColor );
                        
                    }
                     */
                    
                    float const fraction = ( _trail.lastFootstepPositionTime() - _trail.footstepPositionTime( i * 2 ) ) * reciprocalDuration;
                    float const opacity = 1.0f - fraction * maxFractionFactor;
                    
                    drawColor.setA( opacity );
                    insertAtBack( primitive->colors_, drawColor );
                    insertAtBack( primitive->colors_, drawColor );
                    
                    
                    
                    
                    
                    
                    /*
                    float const min = 0.05f;
                    float const fraction = static_cast< float >( i ) * reciprocalTrailLineCount;
                    float const opacity  = ( fraction * ( 1.0f - min ) ) + min;
                    drawColor.setA( opacity );
                    insertAtBack( primitive->colors_, drawColor );
                    insertAtBack( primitive->colors_, drawColor );
                    */
                    
                    
                    
                }  

                
                
                
                
                
                return primitive;
            }
        
        
    } // namespace Graphics

    
} // namespace OpenSteer




#endif //  OPENSTEER_GRAPHICS_OPENGLUTILITIES_H
