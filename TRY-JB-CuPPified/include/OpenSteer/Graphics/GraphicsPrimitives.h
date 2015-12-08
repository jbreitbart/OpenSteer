#ifndef OPENSTEER_GRAPHICS_GRAPICSPRIMITIVES_H
#define OPENSTEER_GRAPHICS_GRAPICSPRIMITIVES_H


// Include std::string
#include <string>

// Include std::vector
#include <vector>

// Include std::pair
#include <utility>

// Include std::auto_ptr
// #include <memory>



// Include OpenSteer::size_t
#include "OpenSteer/StandardTypes.h"

// Include OpenSteer::Color
#include "OpenSteer/Color.h"

// Include OpenSteer::Vec3
#include "OpenSteer/Vec3.h"

// Include OpenSteer::Trail
#include "OpenSteer/Trail.h"



namespace OpenSteer {

    
    namespace Graphics {
        
        // @todo There must be a better solution how to represent and work with
        //       graphics primitives!
        
        // @todo Use covariant return type or an @c auto_ptr for the return type
        //       of @c clone.
        
        class GraphicsPrimitive {
        public:
            typedef size_t size_type;
            typedef Color Material;
            
            virtual ~GraphicsPrimitive() = 0;
            
            virtual Material const& material() const = 0;
            virtual void setMaterial( Material const& _material ) = 0;
            
            /** 
             * Might be necessary to clone in render feeders or would a serialize
             * functionality be better suited because this would be usable with
             * message passing?
             */
            virtual GraphicsPrimitive* clone() const = 0;
            
        }; // class GraphicsPrimitive
        

        
        class CircleGraphicsPrimitive : public GraphicsPrimitive {
        public:
            CircleGraphicsPrimitive() : material_(), radius_( 0.0f ), segmentCount_( 5 ) {}
            CircleGraphicsPrimitive( float const _radius,
                                    Material const& _material,
                                     size_type const _segmentCount ) : material_( _material ), radius_ ( _radius ), segmentCount_ ( _segmentCount ) {}
            virtual ~CircleGraphicsPrimitive() {}
            
            void swap( CircleGraphicsPrimitive& other );
            
            float radius() const {
                return radius_;
            }
            
            
            void setRadius( float _radius ) {
                radius_ = _radius;
            }
            
            size_type segmentCount() const {
                return segmentCount_;
            }
            
            
            void setSegmentCount( size_type _segmentCount ) {
                segmentCount_ = _segmentCount;
            }
            
            virtual Material const& material() const {
                return material_;
            }
            
            
            virtual void setMaterial( Material const& _material ) {
                material_ = _material;
            }
            
            virtual CircleGraphicsPrimitive* clone() const {
                return new CircleGraphicsPrimitive( *this );
            }
            
        private:
            Material material_;
            float radius_;
            size_type segmentCount_;
            
        }; // class CircleGraphicsPrimitive
        
        
        class DiscGraphicsPrimitive : public GraphicsPrimitive {
        public:
            DiscGraphicsPrimitive() : material_(), radius_( 0.0f ), segmentCount_( 5 ) {}
            DiscGraphicsPrimitive( float const _radius,
                                     Material const& _material,
                                     size_type const _segmentCount ) : material_( _material ), radius_ ( _radius ), segmentCount_ ( _segmentCount ) {}
            virtual ~DiscGraphicsPrimitive() {}
            
            void swap( DiscGraphicsPrimitive& other );
            
            float radius() const {
                return radius_;
            }
            
            
            void setRadius( float _radius ) {
                radius_ = _radius;
            }
            
            size_type segmentCount() const {
                return segmentCount_;
            }
            
            
            void setSegmentCount( size_type _segmentCount ) {
                segmentCount_ = _segmentCount;
            }
            
            virtual Material const& material() const {
                return material_;
            }
            
            
            virtual void setMaterial( Material const& _material ) {
                material_ = _material;
            }
            
            virtual DiscGraphicsPrimitive* clone() const {
                return new DiscGraphicsPrimitive( *this );
            }
            
            
        private:
            Material material_;
            float radius_;
            size_type segmentCount_;
            
            
        }; // class DiscGraphicsPrimitive
        
        
        
        
        class LineGraphicsPrimitive : public GraphicsPrimitive {
        public: 
            LineGraphicsPrimitive() : material_(), lineBegin_( 0.0f, 0.0f, 0.0f ), lineEnd_( 0.0f, 0.0f, 0.0f ) {}
            LineGraphicsPrimitive( Vec3 const& _lineBegin, 
                                   Vec3 const& _lineEnd,
                                   Material const& _material ) : material_( _material ), lineBegin_( _lineBegin ), lineEnd_( _lineEnd ) {}
            virtual ~LineGraphicsPrimitive() {}
            
            void swap( LineGraphicsPrimitive& other );
            
            
            void assign( Vec3 const& _lineBegin, 
                         Vec3 const& _lineEnd,
                         Material const& _material  ) {
                lineBegin_ = _lineBegin;
                lineEnd_ = _lineEnd;
                material_ = _material;
            }
            
            virtual Material const& material() const { return material_; }
            virtual void setMaterial( Material const& _material ) { material_ = _material; }
            
            Vec3 const& lineBegin() const { return lineBegin_; }
            void setLineBegin( Vec3 const& _lineBegin ) { lineBegin_ = _lineBegin; }
            
            Vec3 const& lineEnd() const { return lineEnd_; }
            void setLineEnd( Vec3 const& _lineEnd ) { lineEnd_ = _lineEnd; }
            
            virtual LineGraphicsPrimitive* clone() const {
                return new LineGraphicsPrimitive( *this );
            }
            
            
        private:
            Material material_;
            Vec3 lineBegin_;
            Vec3 lineEnd_;
            
        }; // class LineGraphicsPrimitive
        
        
        class Vehicle2dGraphicsPrimitive : public GraphicsPrimitive {
        public:
            Vehicle2dGraphicsPrimitive() : material_( 0.0f, 0.0f, 0.0f ), radius_( 1.0f ) {}
            Vehicle2dGraphicsPrimitive( float _radius, Material const& _material ) : material_( _material ), radius_( _radius ) {}
            virtual ~Vehicle2dGraphicsPrimitive() {}
            
            void swap( Vehicle2dGraphicsPrimitive& other );
            
            void assign(  float _radius, Material const& _material ) {
                radius_ = _radius;
                material_ = _material;
            }
            
            virtual Material const& material() const { return material_; }
            virtual void setMaterial( Material const& _material ) { material_ = _material; }
            
            float radius() const { return radius_; }
            void setRadius( float _radius ) { radius_ = _radius; }
            
            size_t circleSegmentCount() const {
                return 20;
            }
            
            Color circleMaterial() const {
                return gWhite;
            }
            
            virtual Vehicle2dGraphicsPrimitive* clone() const {
                return new Vehicle2dGraphicsPrimitive( *this );
            }
            
            
        private:
            Material material_;
            float radius_;
        };
        
        
		
		
		/**
		 * Defines a vehicle with a triangular arrowhead form.
		 */
		class Basic3dSphericalVehicleGraphicsPrimitive : public GraphicsPrimitive {
		public:
            Basic3dSphericalVehicleGraphicsPrimitive() : material_( 0.0f, 0.0f, 0.0f ), material_variation_factor_( 0.05f ), length_center_( 0.5f ), length_( 0.933f ), width_( 0.5f ), height_( 0.25f ) {}
            Basic3dSphericalVehicleGraphicsPrimitive( float _length_center, float _length, float _width, float _height, Material const& _material, float _material_variation_factor = 0.05f ) : material_( _material ), material_variation_factor_( _material_variation_factor ), length_center_( _length_center ), length_( _length ), width_( _width ), height_( _height ) {}
            virtual ~Basic3dSphericalVehicleGraphicsPrimitive() {}
            
            void swap( Basic3dSphericalVehicleGraphicsPrimitive& other );
            
            void assign( float _length_center, float _length, float _width, float _height, Material const& _material, float _material_variation_factor = 0.05f ) {
                material_ = _material;
				material_variation_factor_ = _material_variation_factor;
				length_center_ = _length_center;
				length_ = _length;
				width_ = _width;
				height_ = _height;
            }
            
            virtual Material const& material() const { return material_; }
            virtual void setMaterial( Material const& _material ) { material_ = _material; }
            
            
			float material_variation_factor() const { return material_variation_factor_; }
			void set_material_variation_factor( float mvf ) { material_variation_factor_ = mvf; }
			
			float length_center() const { return length_center_; }
			void set_length_center( float l ) { length_center_ = l; }
			
			float length() const { return length_; }
			void set_length( float l ) { length_ = l; }
			
			float width() const { return width_; }
			void set_width( float w ) { width_ = w; }
			
			float height() const { return height_; }
			void set_height( float h ) { height_ = h; }
			
            virtual Basic3dSphericalVehicleGraphicsPrimitive* clone() const {
                return new Basic3dSphericalVehicleGraphicsPrimitive( *this );
            }
            
            
		private:
			Material material_;
			float material_variation_factor_;
			// center of mass as seen from the nose to the back of the vehicle.
			float length_center_; 
			float length_;
			float width_;
			float height_; 
        }; // class Basic3dSphericalVehicleGraphicsPrimitive
		
		
        
       
        template< size_t LineCount >
        class TrailLinesGraphicsPrimitive : public GraphicsPrimitive {
        public:
            
            TrailLinesGraphicsPrimitive() : material_( grayColor (0.7f) ), tickMaterial_( gWhite ), trail_(), tickDuration_( 1.0f ) {
                // Nothing to do.
            }
            
            
            TrailLinesGraphicsPrimitive( Material const& _mainMaterial, Material const& _tickMaterial ): material_( _mainMaterial ), tickMaterial_( _tickMaterial ), trail_(), tickDuration_( 0.0f ) {
                // Nothing to do.
            }
            
            TrailLinesGraphicsPrimitive( Material const& _mainMaterial, Material const& _tickMaterial, float _tickDuration ): material_( _mainMaterial ), tickMaterial_( _tickMaterial ), trail_(), tickDuration_( _tickDuration ) {
                // Nothing to do.
            }
            
            explicit TrailLinesGraphicsPrimitive( Trail< LineCount > const& _trail ) : material_( grayColor (0.7f) ), tickMaterial_( gWhite ), trail_( _trail ), tickDuration_( 0.0f ) {
                // Nothing to do.
            }
            
            
            TrailLinesGraphicsPrimitive( Trail< LineCount > const& _trail, Material const& _mainMaterial, Material const& _tickMaterial ): material_( _mainMaterial ), tickMaterial_( _tickMaterial ), trail_( _trail), tickDuration_( 0.0f ) {
                // Nothing to do.
            }
            
            
            TrailLinesGraphicsPrimitive( Trail< LineCount > const& _trail, Material const& _mainMaterial, Material const& _tickMaterial, float _tickDuration ): material_( _mainMaterial ), tickMaterial_( _tickMaterial ), trail_( _trail), tickDuration_( _tickDuration ) {
                // Nothing to do.
            }
            
            
            virtual ~TrailLinesGraphicsPrimitive() {
                // Nothing to do.
            }
            
            void swap( TrailLinesGraphicsPrimitive& _other ) {
                material_.swap( _other.material_ );
                tickMaterial_.swap( _other.tickMaterial_ );
                trail_.swap( _other.trail_ );
            }
            
            void assign( Trail< LineCount > const& _trail, Material const& _mainMaterial, Material const& _tickMaterial ) {
                material_ = _mainMaterial;
                tickMaterial_ = _tickMaterial; 
                trail_ = _trail;
            }
            
            virtual Material const& material() const {
                return material_;
            }
            
            virtual void setMaterial( Color const& _color ) {
                material_ = _color;
            }
            
            
            
            virtual TrailLinesGraphicsPrimitive* clone() const {
                return new TrailLinesGraphicsPrimitive( *this );
            }
            
            
            
            
            
            virtual Material const& tickMaterial() const {
                return tickMaterial_;
            }
            
            virtual void setTickMaterial( Color const& _color ) {
                tickMaterial_ = _color;
            }
            
            
            
            Trail< LineCount > const& trail() const {
                return trail_;
            }
            
            void setTrail( Trail< LineCount > const& _trail ) {
                trail_ = _trail;
            }
            
            void recordPosition( Vec3 const& _position, float _time ) {
                trail_.recordPosition( _position, _time );
            }
            
            
            void clear() {
                trail_.clear();
            }
            
            size_type lineCount() const {
                return trail_.footstepCount();
            }
            
            
            size_type lineVertexCount() const {
                return trail_.positionCount();
            }
            
            
            Vec3 const& vertex( size_type _index ) const {
                return trail_.footstepPosition( _index );
            }
            
            
            bool lineAtTick( size_type _index ) const {
                return trail_.footstepAtTick( _index, tickDuration_ );
            }
            
            float tickDuration() const {
                return tickDuration_;
            }
            
            void setTickDuration( float _tickDuration ) {
                tickDuration_ = _tickDuration;
            }
            
            float duration() const {
                return trail_.duration();
            }
            
            void setDuration( float _duration ) {
                trail_.setDuration( _duration );
            }
            
            
        private:
            Material material_;
            Material tickMaterial_;
            Trail< LineCount > trail_;
            float tickDuration_;
            
        }; // class TrailLinesGraphicsPrimitive
        
        
        class FloorGraphicsPrimitive : public GraphicsPrimitive {
        public:
            // @todo Remove the magic numbers (name them).
            FloorGraphicsPrimitive() : material_( gGray20 ), breadth_( 500 ), length_( 500 ) {
                
            }
            
            FloorGraphicsPrimitive( float _breadth, float _length, Material const& _mainMaterial = Color( gGray20 ) ) : material_( _mainMaterial ), breadth_( _breadth ), length_( _length ) {
                
            }
            
            virtual ~FloorGraphicsPrimitive() {
                // Nothing to do.
            }
            
            void swap( FloorGraphicsPrimitive& _other );
            
            void assign(  float _breadth, float _length, Material const& _material = Color( gGray20 ) ) {
                material_ = _material;
                breadth_ = _breadth;
                length_ = _length;
            }
            
            virtual Material const& material() const { return material_; }
            virtual void setMaterial( Material const& _material ) { material_ = _material; }
            
            
            virtual FloorGraphicsPrimitive* clone() const {
                return new FloorGraphicsPrimitive( *this );
            }
            
            
            
            float breadth() const {
                return breadth_;
            }
            
            void setBreadth( float _breadth ) {
                breadth_ = _breadth;
            }
            
            float length() const {
                return length_;
            }
            
            void setLength( float _length ) {
                length_ = _length;
            }
            
            
        private:
            Material material_;
            float breadth_;
            float length_;
        }; // class FloorGraphicsPrimitive
        
        
        
        
        class TextAt2dLocationGraphicsPrimitive : public GraphicsPrimitive {
        public:
            
            enum RelativePosition { TOP_LEFT, BOTTOM_LEFT, TOP_RIGHT, BOTTOM_RIGHT };
            
            TextAt2dLocationGraphicsPrimitive() : material_(), position_( 0.0f, 0.0f, 0.0f ), relativePosition_( TOP_LEFT ), text_() {}
            TextAt2dLocationGraphicsPrimitive( Vec3 const& _position,
                                               RelativePosition _start,
                                               std::string const& _text,
                                               Material const& _material ) : material_( _material ), position_( _position ), relativePosition_( _start ), text_( _text ) {}
            
            virtual ~TextAt2dLocationGraphicsPrimitive() {
                // Nothing to do.
            }
            
            
            virtual TextAt2dLocationGraphicsPrimitive* clone() const {
                return new TextAt2dLocationGraphicsPrimitive( *this );
            }
            
            
            void swap( TextAt2dLocationGraphicsPrimitive& other );
            
            virtual Material const& material() const {
                return material_;
            }
            
            virtual void setMaterial( Material const& _material ) {
                material_ = _material;
            }
            
            std::string const& text() const {
                return text_;
            }
            
            void setText( std::string const& _text ) {
                text_ = _text;
            }
                 
            Vec3 const& position() const {
                return position_;
            }
            
            void setPosition( Vec3 const& _position ) {
                position_ = _position;
            }
            
            RelativePosition relativePosition() const {
                return relativePosition_;
            }
            
            void setRelativePosition( RelativePosition _start ) {
                relativePosition_ = _start;
            }
            
        private:
            Material material_;
            Vec3 position_;
            RelativePosition relativePosition_;
            std::string text_;
            
        }; // class TextAt2dLocationGraphicsPrimitive
        
        
        
        class TextAt3dLocationGraphicsPrimitive : public GraphicsPrimitive {
        public:
            TextAt3dLocationGraphicsPrimitive() : material_(), text_() {}
            TextAt3dLocationGraphicsPrimitive( std::string const& _text,
                                               Material const& _material ) : material_( _material ), text_( _text ) {}
            
            virtual ~TextAt3dLocationGraphicsPrimitive() {
                // Nothing to do.
            }
            
            
            virtual TextAt3dLocationGraphicsPrimitive* clone() const {
                return new TextAt3dLocationGraphicsPrimitive( *this );
            }
            
            
            void swap( TextAt3dLocationGraphicsPrimitive& other );
            
            virtual Material const& material() const {
                return material_;
            }
            
            virtual void setMaterial( Material const& _material ) {
                material_ = _material;
            }
            
            std::string const& text() const {
                return text_;
            }
            
            void setText( std::string const& _text ) {
                text_ = _text;
            }
            
        private:
            Material material_;
            std::string text_;
            
        }; // class TextAt3dLocationGraphicsPrimitive
        
        
		
		
		class BoxGraphicsPrimitive : public GraphicsPrimitive {
		public:
            BoxGraphicsPrimitive() : material_(), width_( 0.0f ), height_( 0.0f ), depth_( 0.0f ) {}
            BoxGraphicsPrimitive( float const _width,
								  float const _height,
								  float const _depth,
								  Material const& _material) 
				: material_( _material ), width_( _width ), height_( _height ), depth_( _depth ) {}
            virtual ~BoxGraphicsPrimitive() {}
            
            void swap( BoxGraphicsPrimitive& other );
            
			float width() const
			{
				return width_;
			}
            
			float height() const
			{
				return height_;
			}
			
			float depth() const
			{
				return depth_;
			}
            
            void setExtents( float _width, float _height, float _depth ) {
                width_ = _width;
				height_ = _height;
				depth_ = _depth;
            }
                        
            virtual Material const& material() const {
                return material_;
            }
            
            
            virtual void setMaterial( Material const& _material ) {
                material_ = _material;
            }
            
            virtual BoxGraphicsPrimitive* clone() const {
                return new BoxGraphicsPrimitive( *this );
            }
            
		private:
			Material material_;
            float width_;
			float height_;
			float depth_;
            
        }; // class BoxGraphicsPrimitive
		
		
		class SphereGraphicsPrimitive : public GraphicsPrimitive {
		public:
            SphereGraphicsPrimitive() : material_(), radius_( 0.0f ), sliceCount_( 6 ), stackCount_( 6 ) {}
            SphereGraphicsPrimitive( float const _radius,
									 Material const& _material,
                                     size_type const _sliceCount = 6,
									 size_type const _stackCount = 6 ) 
				: material_( _material ), radius_ ( _radius ), sliceCount_( _sliceCount ), stackCount_(_stackCount ) {}
            virtual ~SphereGraphicsPrimitive() {}
            
            void swap( SphereGraphicsPrimitive& other );
            
            float radius() const {
                return radius_;
            }
            
            
            void setRadius( float _radius ) {
                radius_ = _radius;
            }
            
			
			size_type sliceCount() const {
                return sliceCount_;
            }
            
            
            void setSliceCount( size_type _sliceCount ) {
                sliceCount_ = _sliceCount;
			}
			
			
            size_type stackCount() const {
                return stackCount_;
            }
            
            
            void setStackCount( size_type _stackCount ) {
                stackCount_ = _stackCount;
            }
            
            virtual Material const& material() const {
                return material_;
            }
            
            
            virtual void setMaterial( Material const& _material ) {
                material_ = _material;
            }
            
            virtual SphereGraphicsPrimitive* clone() const {
                return new SphereGraphicsPrimitive( *this );
            }
            
		private:
			Material material_;
            float radius_;
            size_type sliceCount_;
			size_type stackCount_;
            
        }; // class SphereGraphicsPrimitive
		
		
		
    } // namespace Graphics
    
    
    
} // namespace OpenSteer

#endif // OPENSTEER_GRAPHICS_GRAPICSPRIMITIVES_H
