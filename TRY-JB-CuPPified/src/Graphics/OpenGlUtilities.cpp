#include "OpenSteer/Graphics/OpenGlUtilities.h"


// Include OpenSteer::sqrtXXX, OpenSteer::square
#include "OpenSteer/Utilities.h"

// Include std::vector
#include <vector>

// Include std::size_t
#include <cstddef>

// Include kapaga::pi<T>()
#include "kapaga/math_utilities.h"

OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlImage > 
OpenSteer::Graphics::makeCheckerboardImage(  size_t _width, size_t _height, size_t _subdivisions, Color const& _color0, Color const& _color1 ) 
{    
    assert( 0 != _subdivisions && "At least one subdivision is needed.");
    
    // @todo Add asserts that the dimensions are powers of
    //       the base @c 2.
    
    SharedPointer< OpenGlImage > image( new OpenGlImage( _width, _height ) );
    
    size_t const checkSizeWidth = _width / _subdivisions;
    size_t const checkSizeHeight = _height / _subdivisions;
    
    // Algorithm from Woo, et al, OpenGL Programming Guide - Third Edition,
    // Addison-Wesley 1999, pp. 358--360.
    for ( size_t height = 0; height != _width; ++height ) {
        for ( size_t width = 0; width != _height; ++width ) {            
            bool const alternate = static_cast< bool >( ( ( ( height & checkSizeHeight ) == 0 ) ^ ( ( width & checkSizeWidth ) ) == 0 ) );
            
            if ( alternate ) {
                image->setPixel( width, height, makePixelColor( _color0 ) );
            } else {
                image->setPixel( width, height, makePixelColor( _color1 ) );
            }
        }
    }
    
    return image;
}


OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderMesh >
OpenSteer::Graphics::createLine( Vec3 const& lineBegin, Vec3 const& lineEnd, Color const& color ) 
{
    OpenSteer::SharedPointer< OpenGlRenderMesh > primitive( new OpenGlRenderMesh( OpenGlRenderMesh::LINES, 2 ) );
    // Each Vec3 has three float elements.
    insertAtBack( primitive->vertices_, lineBegin );
    insertAtBack( primitive->vertices_, lineEnd );
    
    // Each color has four float elements.
    insertAtBack( primitive->colors_, color );
    insertAtBack( primitive->colors_, color );
    
    // @todo Calculate an orthogonal up vector.
    Vec3 const up( findPerpendicularIn3d( lineEnd - lineBegin ) );
    
    insertAtBack( primitive->normals_, up );
    insertAtBack( primitive->normals_, up );
    
    primitive->indices_.push_back( 0 );
    primitive->indices_.push_back( 1 ); 
    
    return primitive;
}


OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderMesh >
OpenSteer::Graphics::createVehicleTriangle( float radius, Color const& color ) 
{
    OpenSteer::SharedPointer< OpenGlRenderMesh > primitive( new OpenGlRenderMesh( OpenGlRenderMesh::TRIANGLES, 3 ) );
    
    {
        float const x = 0.5f;
        float const y = sqrtXXX( 1 - square( x ) );
        
        // @todo Don't push the vehicle up!
        Vec3 const u = radius * 0.05f * Vec3( 0.0f, 1.0f, 0.0f );
        Vec3 const f = radius * Vec3( 0.0f, 0.0f, 1.0f );
        Vec3 const s = radius * x * Vec3( 1.0f, 0.0f, 0.0f );
        Vec3 const b = radius * y * Vec3( 0.0f, 0.0f, -1.0f );
        
        Vec3 const first = f + u;
        Vec3 const second = b - s + u;
        Vec3 const third = b  + s + u;
        
        // Each Vec3 has three float elements.
        insertAtBack( primitive->vertices_, first );
        insertAtBack( primitive->vertices_, second );
        insertAtBack( primitive->vertices_, third );
    }
    
    // Each color has four float elements.
    insertAtBack( primitive->colors_, color );
    insertAtBack( primitive->colors_, color );
    insertAtBack( primitive->colors_, color );    
    
    insertAtBack( primitive->normals_, Vec3::up );
    insertAtBack( primitive->normals_, Vec3::up );
    insertAtBack( primitive->normals_, Vec3::up );
    
    primitive->indices_.push_back( 0 );
    primitive->indices_.push_back( 1 );
    primitive->indices_.push_back( 2 );
    
    return primitive;
}




OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderMesh > 
OpenSteer::Graphics::createBasic3dSphericalVehicle( float length_center,
													float length,
													float width,
													float height, 
													Color const& color,
													float material_variation_factor )
{
    OpenSteer::SharedPointer< OpenGlRenderMesh > primitive( new OpenGlRenderMesh( OpenGlRenderMesh::TRIANGLES, 18 ) );
    	
	// Create triangle vertices	
	Vec3 const nose = Vec3( 0.0f, 0.0f, length_center );
	Vec3 const side_right = Vec3( width / 2.0f, 0.0f, length_center - length );
	Vec3 const side_left = Vec3( -width / 2.0f, 0.0f, length_center - length );
	Vec3 const top = Vec3( 0.0f, height / 2.0f , length_center - length );
	Vec3 const bottom = Vec3( 0.0f, - height / 2.0f , length_center - length );
	
	// Each Vec3 has three float elements.
	// Vertices in counter-clock wise ordering describe a front-facing triangle.
	// top, right
	insertAtBack( primitive->vertices_, nose );
	insertAtBack( primitive->vertices_, side_right );
	insertAtBack( primitive->vertices_, top );
	// top, left
	insertAtBack( primitive->vertices_, nose );
	insertAtBack( primitive->vertices_, top );
	insertAtBack( primitive->vertices_, side_left );
	// bottom, right
	insertAtBack( primitive->vertices_, nose );
	insertAtBack( primitive->vertices_, bottom );
	insertAtBack( primitive->vertices_, side_right );
	// bottom, left
	insertAtBack( primitive->vertices_, nose );
	insertAtBack( primitive->vertices_, side_left );
	insertAtBack( primitive->vertices_, bottom );
	// back, right
	insertAtBack( primitive->vertices_, bottom );
	insertAtBack( primitive->vertices_, top );
	insertAtBack( primitive->vertices_, side_right );
	// back, left
	insertAtBack( primitive->vertices_, bottom );
	insertAtBack( primitive->vertices_, side_left );
	insertAtBack( primitive->vertices_, top );
	
	
	// Create normals for the triangles and attach them to the vertices.
	Vec3 const top_right_normal( crossProduct( side_right - nose, top - nose ).normalize() );
	Vec3 const top_left_normal( crossProduct( top - nose, side_left - nose ).normalize() );
	Vec3 const bottom_right_normal( crossProduct( bottom - nose, side_right - nose ).normalize() );
	Vec3 const bottom_left_normal( crossProduct( side_left - nose, bottom - nose ).normalize() );
	Vec3 const back_right_normal( crossProduct( top - side_right, bottom - side_right ).normalize() );
	Vec3 const back_left_normal( crossProduct( side_left - top, bottom - top ).normalize() );
	
	// top, right
	insertAtBack( primitive->normals_, top_right_normal );
	insertAtBack( primitive->normals_, top_right_normal );
	insertAtBack( primitive->normals_, top_right_normal );
	// top, left
	insertAtBack( primitive->normals_, top_left_normal );
	insertAtBack( primitive->normals_, top_left_normal );
	insertAtBack( primitive->normals_, top_left_normal );
	// bottom, right
	insertAtBack( primitive->normals_, bottom_right_normal );
	insertAtBack( primitive->normals_, bottom_right_normal );
	insertAtBack( primitive->normals_, bottom_right_normal );
	// bottom, left
	insertAtBack( primitive->normals_, bottom_left_normal );
	insertAtBack( primitive->normals_, bottom_left_normal );
	insertAtBack( primitive->normals_, bottom_left_normal );
	// back, right
	insertAtBack( primitive->normals_, back_right_normal );
	insertAtBack( primitive->normals_, back_right_normal );
	insertAtBack( primitive->normals_, back_right_normal );
	// back, left
	insertAtBack( primitive->normals_, back_left_normal );
	insertAtBack( primitive->normals_, back_left_normal );
	insertAtBack( primitive->normals_, back_left_normal );
    
	
	
	// Create colors for the vertices.
	float const j = material_variation_factor;
	float const k = -j;
	Color const top_right_color( color + Color( j, j, k ) );
	Color const top_left_color( color + Color( j, k, j ) );
	Color const bottom_right_color( color + Color( k, j, j ) );
	Color const bottom_left_color( color + Color( k, j, k ) );
	Color const back_right_color( color + Color( k, k, j ) );
	Color const back_left_color( color + Color( k, k, j ) );
	
	// top, right
	insertAtBack( primitive->colors_, top_right_color );
	insertAtBack( primitive->colors_, top_right_color );
	insertAtBack( primitive->colors_, top_right_color );
	// top, left
	insertAtBack( primitive->colors_, top_left_color );
	insertAtBack( primitive->colors_, top_left_color );
	insertAtBack( primitive->colors_, top_left_color );
	// bottom, right
	insertAtBack( primitive->colors_, bottom_right_color );
	insertAtBack( primitive->colors_, bottom_right_color );
	insertAtBack( primitive->colors_, bottom_right_color );
	// bottom, left
	insertAtBack( primitive->colors_, bottom_left_color );
	insertAtBack( primitive->colors_, bottom_left_color );
	insertAtBack( primitive->colors_, bottom_left_color );
	// back, right
	insertAtBack( primitive->colors_, back_right_color );
	insertAtBack( primitive->colors_, back_right_color );
	insertAtBack( primitive->colors_, back_right_color );
	// back, left
	insertAtBack( primitive->colors_, back_left_color );
	insertAtBack( primitive->colors_, back_left_color );
	insertAtBack( primitive->colors_, back_left_color );
	
	
	// Create the indices for the vertices, normals, colors.
	// top, right
    primitive->indices_.push_back( 0 );
    primitive->indices_.push_back( 1 );
    primitive->indices_.push_back( 2 );
	// top, left
    primitive->indices_.push_back( 3 );
    primitive->indices_.push_back( 4 );
    primitive->indices_.push_back( 5 );
	// bottom, right
    primitive->indices_.push_back( 6 );
    primitive->indices_.push_back( 7 );
    primitive->indices_.push_back( 8 );
	// bottom, left
    primitive->indices_.push_back( 9 );
    primitive->indices_.push_back( 10 );
    primitive->indices_.push_back( 11 );
	// back, right
    primitive->indices_.push_back( 12 );
    primitive->indices_.push_back( 13 );
    primitive->indices_.push_back( 14 );
	// back, left
    primitive->indices_.push_back( 15 );
    primitive->indices_.push_back( 16 );
    primitive->indices_.push_back( 17 );
    
    return primitive;	
}





OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderMesh >
OpenSteer::Graphics::createCircle( float radius, size_t segmentCount, Color const& color )
{
    OpenSteer::SharedPointer< OpenGlRenderMesh > primitive( new OpenGlRenderMesh( OpenGlRenderMesh::LINE_LOOP, segmentCount ) );
    
    Vec3 pointOnCircle( radius, 0.0f, 0.0f );
    float const step = ( 2 * OPENSTEER_M_PI ) / segmentCount;
    float sin = 0;
    float cos = 0;
    for ( size_t i = 0; i != segmentCount; ++i ) {
        insertAtBack( primitive->vertices_, pointOnCircle );
        insertAtBack( primitive->colors_, color );
        insertAtBack( primitive->normals_, Vec3::up );
        primitive->indices_.push_back( i );
        
        // @todo Profile if this is a performance bottleneck because of the 
        //       @c if clause used to test @c sin and @c cos for @c 0.
        pointOnCircle = pointOnCircle.rotateAboutGlobalY( step, sin, cos );
    }
    
    
    return primitive;
}


OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderMesh >
OpenSteer::Graphics::createDisc( float radius, size_t segmentCount, Color const& color ) 
{
    OpenSteer::SharedPointer< OpenGlRenderMesh > primitive( new OpenGlRenderMesh( OpenGlRenderMesh::TRIANGLE_FAN, segmentCount ) );
    
    // First enter the center into the primitive.
    insertAtBack( primitive->vertices_, Vec3( 0.0f, 0.0f, 0.0f ) );
    insertAtBack( primitive->colors_, color );
    insertAtBack( primitive->normals_, Vec3::up );
    primitive->indices_.push_back( 0 );
    
    
    Vec3 pointOnCircle( radius, 0.0f, 0.0f );
    float const step = ( 2 * OPENSTEER_M_PI ) / segmentCount;
    float sin = 0;
    float cos = 0;
    for ( size_t i = 0; i != segmentCount; ++i ) {
        insertAtBack( primitive->vertices_, pointOnCircle );
        insertAtBack( primitive->colors_, color );
        insertAtBack( primitive->normals_, Vec3::up );
        primitive->indices_.push_back( i + 1 );
        
        // @todo Profile if this is a performance bottleneck because of the 
        //       @c if clause used to test @c sin and @c cos for @c 0.
        pointOnCircle = pointOnCircle.rotateAboutGlobalY( step, sin, cos );
    }
    
    // Last enter the first point on the radius again to close the tri fan.
    insertAtBack( primitive->vertices_, Vec3( radius, 0.0f, 0.0f ) );
    insertAtBack( primitive->colors_, color );
    insertAtBack( primitive->normals_, Vec3::up );
    primitive->indices_.push_back( segmentCount + 1 );
    
    
    return primitive; 
}



OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderMesh >
OpenSteer::Graphics::createFloor( float _broadth, 
                                  float _length, 
                                  Color const& _color, 
                                  OpenGlRenderMesh::TextureId _textureId,
                                  float _broadthTextureScale,
                                  float _lengthTextureScale )
{
    OpenSteer::SharedPointer< OpenGlRenderMesh > primitive( new OpenGlRenderMesh( OpenGlRenderMesh::QUADS, 4 ) );
    
    // @todo Put all the calculated data first into a const array and insert
    //       this into the render mesh?
    
    insertAtBack( primitive->vertices_, Vec3( -_broadth * 0.5f, -0.1f,  _length * 0.5f ) );
    insertAtBack( primitive->vertices_, Vec3(  _broadth * 0.5f, -0.1f,  _length * 0.5f ) );
    insertAtBack( primitive->vertices_, Vec3(  _broadth * 0.5f, -0.1f, -_length * 0.5f ) );
    insertAtBack( primitive->vertices_, Vec3( -_broadth * 0.5f, -0.1f, -_length * 0.5f ) );
    
    insertAtBack( primitive->colors_, _color );
    insertAtBack( primitive->colors_, _color );
    insertAtBack( primitive->colors_, _color );
    insertAtBack( primitive->colors_, _color );
    
    insertAtBack( primitive->normals_, Vec3::up );
    insertAtBack( primitive->normals_, Vec3::up );
    insertAtBack( primitive->normals_, Vec3::up );
    insertAtBack( primitive->normals_, Vec3::up );
    
    primitive->textureId_ = _textureId;
    primitive->textureFunction_ = OpenGlRenderMesh::REPLACE;
    
    // @todo Add a way to adapt to the texture image!
    
    primitive->textureCoordinates_.push_back( 0.0f );
    primitive->textureCoordinates_.push_back( 0.0f );
    primitive->textureCoordinates_.push_back( 1.0f * _broadthTextureScale );
    primitive->textureCoordinates_.push_back( 0.0f );
    primitive->textureCoordinates_.push_back( 1.0f * _broadthTextureScale );
    primitive->textureCoordinates_.push_back( 1.0f * _lengthTextureScale );
    primitive->textureCoordinates_.push_back( 0.0f );
    primitive->textureCoordinates_.push_back( 1.0f * _lengthTextureScale );
    
    primitive->indices_.push_back( 0 );
    primitive->indices_.push_back( 1 );
    primitive->indices_.push_back( 2 );
    primitive->indices_.push_back( 3 );
    
    return primitive;
}




OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderMesh > 
OpenSteer::Graphics::createBox( float width, 
								float height, 
								float depth, 
								Color const& color )
{
	OpenSteer::SharedPointer< OpenGlRenderMesh > primitive( new OpenGlRenderMesh( OpenGlRenderMesh::TRIANGLES, 36 ) );
	
	primitive->polygonMode_ = OpenGlRenderMesh::LINE;

	// Extents of the box
	float const w = width / 2.0f;
	float const h = height / 2.0f;
	float const d = depth / 2.0f;

	
	// Each Vec3 has three float elements.
	// Vertices in counter-clock wise ordering describe a front-facing triangle.
	// front
	insertAtBack( primitive->vertices_, Vec3( -w, -h,  d ) );
	insertAtBack( primitive->vertices_, Vec3(  w, -h,  d ) );
	insertAtBack( primitive->vertices_, Vec3(  w,  h,  d ) );
	
	insertAtBack( primitive->vertices_, Vec3(  w,  h,  d ) );
	insertAtBack( primitive->vertices_, Vec3( -w,  h,  d ) );
	insertAtBack( primitive->vertices_, Vec3( -w, -h,  d ) );
	
	// back
	insertAtBack( primitive->vertices_, Vec3(  w, -h, -d ) );
	insertAtBack( primitive->vertices_, Vec3( -w, -h, -d ) );
	insertAtBack( primitive->vertices_, Vec3( -w,  h, -d ) );
	insertAtBack( primitive->vertices_, Vec3( -w,  h, -d ) );
	insertAtBack( primitive->vertices_, Vec3(  w,  h, -d ) );
	insertAtBack( primitive->vertices_, Vec3(  w, -h, -d ) );

	// left
	insertAtBack( primitive->vertices_, Vec3( -w, -h, -d ) );
	insertAtBack( primitive->vertices_, Vec3( -w, -h,  d ) );
	insertAtBack( primitive->vertices_, Vec3( -w,  h,  d ) );
	insertAtBack( primitive->vertices_, Vec3( -w,  h,  d ) );
	insertAtBack( primitive->vertices_, Vec3( -w,  h, -d ) );
	insertAtBack( primitive->vertices_, Vec3( -w, -h, -d ) );
	
	// right
	insertAtBack( primitive->vertices_, Vec3(  w, -h,  d ) );
	insertAtBack( primitive->vertices_, Vec3(  w, -h, -d ) );
	insertAtBack( primitive->vertices_, Vec3(  w,  h, -d ) );
	insertAtBack( primitive->vertices_, Vec3(  w,  h, -d ) );
	insertAtBack( primitive->vertices_, Vec3(  w,  h,  d ) );
	insertAtBack( primitive->vertices_, Vec3(  w, -h,  d ) );
	
	// bottom
	insertAtBack( primitive->vertices_, Vec3( -w, -h, -d ) );
	insertAtBack( primitive->vertices_, Vec3(  w, -h, -d ) );
	insertAtBack( primitive->vertices_, Vec3(  w, -h,  d ) );
	insertAtBack( primitive->vertices_, Vec3(  w, -h,  d ) );
	insertAtBack( primitive->vertices_, Vec3( -w, -h,  d ) );
	insertAtBack( primitive->vertices_, Vec3( -w, -h, -d ) );
	
	// top
	insertAtBack( primitive->vertices_, Vec3( -w,  h,  d ) );
	insertAtBack( primitive->vertices_, Vec3(  w,  h,  d ) );
	insertAtBack( primitive->vertices_, Vec3(  w,  h, -d ) );
	insertAtBack( primitive->vertices_, Vec3(  w,  h, -d ) );
	insertAtBack( primitive->vertices_, Vec3( -w,  h, -d ) );
	insertAtBack( primitive->vertices_, Vec3( -w,  h,  d ) );
	
	

	
	
	// Create normals for the triangles and attach them to the vertices.
	// front
	insertAtBack( primitive->normals_, Vec3( 0.0f, 0.0f, 1.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 0.0f, 1.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 0.0f, 1.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 0.0f, 1.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 0.0f, 1.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 0.0f, 1.0f ) );
	
	// back
	insertAtBack( primitive->normals_, Vec3( 0.0f, 0.0f, -1.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 0.0f, -1.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 0.0f, -1.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 0.0f, -1.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 0.0f, -1.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 0.0f, -1.0f ) );	
	// left
	insertAtBack( primitive->normals_, Vec3( -1.0f, 0.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( -1.0f, 0.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( -1.0f, 0.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( -1.0f, 0.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( -1.0f, 0.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( -1.0f, 0.0f, 0.0f ) );	
	// right
	insertAtBack( primitive->normals_, Vec3( 1.0f, 0.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 1.0f, 0.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 1.0f, 0.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 1.0f, 0.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 1.0f, 0.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 1.0f, 0.0f, 0.0f ) );	
	// bottom
	insertAtBack( primitive->normals_, Vec3( 0.0f, -1.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, -1.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, -1.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, -1.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, -1.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, -1.0f, 0.0f ) );	
	// top
	insertAtBack( primitive->normals_, Vec3( 0.0f, 1.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 1.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 1.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 1.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 1.0f, 0.0f ) );
	insertAtBack( primitive->normals_, Vec3( 0.0f, 1.0f, 0.0f ) );    
	
	
	// Create colors for the vertices.
	// front
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	
	// back
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
		
	// left
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	
	// right
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	
	// bottom
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	
	// top
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	insertAtBack( primitive->colors_, color );
	
	
	// Create the indices for the vertices, normals, colors.
	// front
    primitive->indices_.push_back( 0 );
	primitive->indices_.push_back( 1 );
	primitive->indices_.push_back( 2 );
	primitive->indices_.push_back( 3 );
	primitive->indices_.push_back( 4 );
	primitive->indices_.push_back( 5 );

	// back
	primitive->indices_.push_back( 6 );
	primitive->indices_.push_back( 7 );
	primitive->indices_.push_back( 8 );
	primitive->indices_.push_back( 9 );
	primitive->indices_.push_back( 10 );
	primitive->indices_.push_back( 11 );
	
	// left
	primitive->indices_.push_back( 12 );
	primitive->indices_.push_back( 13 );
	primitive->indices_.push_back( 14 );
	primitive->indices_.push_back( 15 );
	primitive->indices_.push_back( 16 );
	primitive->indices_.push_back( 17 );
	
	// right
	primitive->indices_.push_back( 18 );
	primitive->indices_.push_back( 19 );
	primitive->indices_.push_back( 20 );
	primitive->indices_.push_back( 21 );
	primitive->indices_.push_back( 22 );
	primitive->indices_.push_back( 23 );
	
	// bottom
	primitive->indices_.push_back( 24 );
	primitive->indices_.push_back( 25 );
	primitive->indices_.push_back( 26 );
	primitive->indices_.push_back( 27 );
	primitive->indices_.push_back( 28 );
	primitive->indices_.push_back( 29 );
	
	// top
	primitive->indices_.push_back( 30 );
	primitive->indices_.push_back( 31 );
	primitive->indices_.push_back( 32 );
	primitive->indices_.push_back( 33 );
	primitive->indices_.push_back( 34 );
	primitive->indices_.push_back( 35 );
	
    
    return primitive;									
								
}



OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderMesh > 
OpenSteer::Graphics::createSphere( float const radius, 
								   size_t const slices,
								   size_t const stacks, 
								   Color const& color )
{
	assert( radius >= 0 &&  "radius must be positive or zero." );
	assert( slices > 0 && "For drawing a sphere at least 1 slice is necessary." );
	assert( stacks > 0 && "For drawing a sphere at least 1 stack is necessary." );

		
	// Coords of the quads that approximate a sphere using @a slices slices and 
	// @a stacks stacks.
	// (Sliced) stacks are stored one after the other.
	std::size_t const number_of_coords = ( slices + 1 ) * ( stacks + 1 );
	std::vector< Vec3 > coords;
	coords.reserve( number_of_coords );
		
	float stack_angle = 0.0f;

	float const stack_angle_increment = kapaga::pi<float>() / stacks;
	float const slice_angle_increment = 2.0f * kapaga::pi<float>() / slices;
	
	// Create all coords needed for the sphere. A number of stack coords is
	// duplicated (where the sliced circle closes).
	for ( size_t stack_index = 0; stack_index <= stacks; ++stack_index ) {
	
		float const height = radius * cos( stack_angle );
		// Orthogonal distance to the z-axis at @c height.
		float const width = radius * sin( stack_angle );
		
		float slice_angle = 0.0f;
		
		for ( size_t slice_index = 0; slice_index <= slices; ++slice_index ) {
			Vec3 coord(  width * cos( slice_angle ), height, -width * sin( slice_angle ) );
			
			// std::cout << coord << " ";
			
			coords.push_back( coord );
		
			slice_angle += slice_angle_increment;
		}
	
		stack_angle += stack_angle_increment;
	}
	
	
	// Create the mesh
	OpenSteer::SharedPointer< OpenGlRenderMesh > primitive( new OpenGlRenderMesh( OpenGlRenderMesh::QUADS, stacks * slices * 4 ) );								   
	primitive->polygonMode_ = OpenGlRenderMesh::LINE;
	
	for ( size_t stack_index = 1; stack_index <= stacks; ++stack_index ) {
	
		for ( size_t slice_index = 1; slice_index <= slices; ++slice_index ) {
			
			
			std::size_t const index_a = ( stack_index * ( slices + 1 ) ) + slice_index - 1;
			std::size_t const index_b = ( stack_index * ( slices + 1 ) ) + slice_index;
			std::size_t const index_c = ( ( stack_index - 1 ) * ( slices + 1 ) ) + slice_index;
			std::size_t const index_d = ( ( stack_index - 1 ) * ( slices + 1 ) ) + slice_index - 1;
			
			Vec3 const a( coords[ index_a ] );
			Vec3 const b( coords[ index_b ] );
			Vec3 const c( coords[ index_c ] );
			Vec3 const d( coords[ index_d ] );
			
			// vertices
			insertAtBack( primitive->vertices_, a );
			insertAtBack( primitive->vertices_, b );
			insertAtBack( primitive->vertices_, c );
			insertAtBack( primitive->vertices_, d );
			
			// normals
			Vec3 normal;
			// Correct normals at the lower pole because two coordinates are nearly equal.
			if ( stack_index < stacks ) {
				normal.cross( b - a, d - a );
			} else {
				normal.cross( d - c, b - c );
			}
			normal.normalize();
			
			insertAtBack( primitive->normals_, normal ); 
			insertAtBack( primitive->normals_, normal ); 
			insertAtBack( primitive->normals_, normal ); 
			insertAtBack( primitive->normals_, normal ); 
			
			// color
			insertAtBack( primitive->colors_, color );
			insertAtBack( primitive->colors_, color );
			insertAtBack( primitive->colors_, color );
			insertAtBack( primitive->colors_, color );

			
			// indices
			std::size_t const index_counter = ( ( stack_index - 1 ) * slices * 4 ) + ( slice_index - 1 ) * 4;
			primitive->indices_.push_back( index_counter );
			primitive->indices_.push_back( index_counter + 1 );
			primitive->indices_.push_back( index_counter + 2 );
			primitive->indices_.push_back( index_counter + 3 );
			
		}
	}

	return primitive;
}





