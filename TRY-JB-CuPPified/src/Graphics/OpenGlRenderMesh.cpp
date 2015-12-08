#include "OpenSteer/Graphics/OpenGlRenderMesh.h"

// Include std::cout, std::endl
#include <iostream>

// Include std::for_each
#include <algorithm>

// Include std::advance
#include <iterator>

// Include assert
#include <cassert>

// Include std::bind2nd
#include <functional>



// Include OpenSteer::shrinkToFit
#include "OpenSteer/Utilities.h"


void 
OpenSteer::Graphics::OpenGlRenderMesh::clear()
{
    vertices_.clear();
    colors_.clear();
    normals_.clear();
    textureCoordinates_.clear();
    indices_.clear();
}



void 
OpenSteer::Graphics::OpenGlRenderMesh::shrinkContainersToFit()
{
    shrinkToFit( vertices_ );
    shrinkToFit( colors_ );
    shrinkToFit( normals_ );
    shrinkToFit( textureCoordinates_ );
    shrinkToFit( indices_ );
}



void 
OpenSteer::Graphics::OpenGlRenderMesh::clearAndShrinkContainersToFit()
{
    clear();
    shrinkContainersToFit();
}




namespace {
    
    
    void printFloats( float t ) {
        std::cout << t << " ";
    }
    
    void printUnsignedInts( unsigned int t ) {
        std::cout << t << " ";
    }
    
} // namespace anonymous

void 
OpenSteer::Graphics::print( OpenGlRenderMesh const& primitive )
{
    std::cout << "OpenGlRenderMesh" << std::endl;
    std::cout << "vertices ";
    std::for_each( primitive.vertices_.begin(), primitive.vertices_.end(), printFloats );
    std:: cout << std::endl << "colors ";
    std::for_each( primitive.colors_.begin(), primitive.colors_.end(), printFloats );
    std:: cout << std::endl << "normals ";
    std::for_each( primitive.normals_.begin(), primitive.normals_.end(), printFloats );
    std:: cout << std::endl << "texture coordinates ";
    std::for_each( primitive.textureCoordinates_.begin(), primitive.textureCoordinates_.end(), printFloats );
    std::cout << std::endl << "indices ";
    std::for_each( primitive.indices_.begin(), primitive.indices_.end(), printUnsignedInts );
    std::cout << std::endl;
    
}

/* @todo Remove - has been inlined.
bool 
OpenSteer::Graphics::textured( OpenGlRenderMesh const& _mesh )
{
    // An OpenGL texture id of @c 0 descibes the default texture - which is
    // interpreted as if @a _mesh is untextured.
    return 0 != _mesh.textureId_;
}
*/




/* @todo Remove - has been inlined.
void 
OpenSteer::Graphics::insertAtBack( std::vector< float >& _container, Vec3 const& _data )
{
    _container.insert( _container.end(), _data.data(), _data.data() + 3 );
}
*/

/* @todo Remove - has been inlined.
void 
OpenSteer::Graphics::insertAtBack( std::vector< float >& _colors, Color const& _color )
{
    _colors.insert( _colors.end(), _color.colorFloatArray(), _color.colorFloatArray() + OpenGlRenderMesh::colorElementCount_ );
}
*/



void 
OpenSteer::Graphics::mergeMeshesFast( OpenGlRenderMesh& _target, OpenGlRenderMesh const& _source )
{
    
    
    assert( ( _source.type_ == _target.type_ ) && "_target and _source type must be the same." );
    assert( (_source.textureId_ == _target.textureId_ ) && "_target and _source must have the same textureId to be merged." );
    
    _target.vertices_.insert( _target.vertices_.end(), _source.vertices_.begin(), _source.vertices_.end() );
    _target.colors_.insert( _target.colors_.end(), _source.colors_.begin(), _source.colors_.end() );
    _target.normals_.insert( _target.normals_.end(), _source.normals_.begin(), _source.normals_.end() );
    _target.textureCoordinates_.insert( _target.textureCoordinates_.end(), 
                                       _source.textureCoordinates_.begin(), 
                                       _source.textureCoordinates_.end() );
    
    // Add the indices and adapt them to the index of the added vertices, colors, etc.
    typedef OpenGlRenderMesh::size_type size_type;
    size_type startIndex = _target.indices_.size();
    _target.indices_.resize( startIndex + _source.indices_.size() );
    typedef std::vector< GLuint >::iterator iterator;
    iterator targetIndicesIter = _target.indices_.begin();
    std::advance( targetIndicesIter, startIndex );
    std::transform( _source.indices_.begin(), 
                    _source.indices_.end(), 
                    targetIndicesIter,
                    std::bind2nd( std::plus< GLuint >(), startIndex ) );

    
}


bool 
OpenSteer::Graphics::mergeMeshes( OpenGlRenderMesh& _target, OpenGlRenderMesh const& _source )
{
    if ( ( _source.type_ != _target.type_ ) ||
         ( _source.textureId_ != _target.textureId_ ) ) {
        return false;
    }
    
    mergeMeshesFast( _target, _source );
    return true;
}








