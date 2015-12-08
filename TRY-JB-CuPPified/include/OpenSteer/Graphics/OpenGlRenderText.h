/**
 * Plain old datatypes (PODs) to be feed into the OpenGLRenderer. Classes have 
 * no won behavior other than constructors.
 */
#ifndef OPENSTEER_GRAPHICS_OPENGLRENDERTEXT_H
#define OPENSTEER_GRAPHICS_OPENGLRENDERTEXT_H

// Include std::string
#include <string>


// Include OpenSteer::Vec3
#include "OpenSteer/Vec3.h"

// Include OpenSteer::Color
#include "OpenSteer/Color.h"


namespace OpenSteer {
    
    namespace Graphics {
        
        
        class OpenGlRenderText {
        public:
            
            OpenGlRenderText() : text_(), material_() { /* Nothing to do. */ }
            OpenGlRenderText( std::string const& _text ) : text_( _text ), material_() { /* Nothing to do. */ }
            OpenGlRenderText( std::string const& _text, Color const& _material ): text_( _text ), material_( _material ) {  /* Nothing to do. */  }
            
            std::string text_;
            Color material_;
        }; // struct OpenGlRenderText
        
        
        class OpenGlRenderText2d : private OpenGlRenderText {
        public:
            enum RelativePosition { TOP_LEFT, BOTTOM_LEFT, TOP_RIGHT, BOTTOM_RIGHT };
            
            typedef OpenSteer::size_t size_type;
            
            OpenGlRenderText2d( std::string const& _text ) : OpenGlRenderText( _text ), position_( 0.0f, 0.0f, 0.0f ), relativePosition_( TOP_LEFT ) { /* Nothing to do. */ }
            OpenGlRenderText2d( Vec3 const& _position, RelativePosition _relativePosition ) : OpenGlRenderText(), position_( _position ), relativePosition_( _relativePosition ) { /* Nothing to do. */ }
            OpenGlRenderText2d( Vec3 const& _position, RelativePosition _relativePosition, std::string const& _text ) : OpenGlRenderText( _text ), position_( _position ), relativePosition_( _relativePosition ) { /* Nothing to do. */ }
            OpenGlRenderText2d( Vec3 const& _position, RelativePosition _relativePosition, std::string const& _text, Color const& _material ) : OpenGlRenderText( _text, _material ), position_( _position ), relativePosition_( _relativePosition ) { /* Nothing to do. */ }
            
            
            using OpenGlRenderText::text_;
            using OpenGlRenderText::material_;
            
            Vec3 position_;
            RelativePosition relativePosition_;
            
        }; // class OpenGlRenderText2d
        
        
    } // namespace Graphics
    
    
} // namespace OpenSteer



#endif // OPENSTEER_GRAPHICS_OPENGLRENDERTEXT_H
