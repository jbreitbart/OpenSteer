#include "OpenSteer/Graphics/OpenGlRenderService.h"

// Include std::unary_function
#include <functional>

// Include std::for_each, std::find_if, std::transform
#include <algorithm>

// Include std::back_inserter
#include <iterator>

// Include assert
#include <cassert>

// Inlcude std::numeric_limits<>::max
#include <limits>



namespace {
    using namespace OpenSteer;
    using namespace OpenSteer::Graphics;
    
    
    OpenSteer::size_t const checkerboardImageWidth = 64;
    OpenSteer::size_t const checkerboardImageHeight = checkerboardImageWidth;
    
    
    
    
    /**
     * Functor adding a given instance id to a specified renderer for later 
     * rendering.
     */
    class AddUntransformedInstanceToRender : public std::unary_function< OpenGlRenderer::InstanceId, void > {
    public:
        AddUntransformedInstanceToRender( OpenGlRenderer* _renderer ) : renderer_( _renderer ) {
        }
        
        void operator()( OpenGlRenderer::InstanceId const& _instanceId  ) {
            renderer_->addToRender( _instanceId );
        }
        
    private:
        OpenGlRenderer* renderer_;
    }; // AddUntransformedInstanceToRender
    
    
    
    /**
     * Functor adding a given instance id to a specified renderer for later 
     * rendering with a specified transformation.
     */
    class AddTransformedInstanceToRender : public std::unary_function< OpenGlRenderer::InstanceId, void > {
    public:   
        AddTransformedInstanceToRender( Matrix const& _transformation, OpenGlRenderer* _renderer ) : transformation_( _transformation ), renderer_( _renderer ) {
        }
        
        void operator()( OpenGlRenderer::InstanceId const& _instanceId  ) {
            renderer_->addToRender( transformation_,  _instanceId );
        }
    private:
        Matrix transformation_;
        OpenGlRenderer* renderer_;
        
    }; // AddTransformedInstanceToRender
    
    
    /**
     * Functor removing a given instance id from a specified renderer.
     */
    class RemoveInstanceFromRenderMeshLibrary : public std::unary_function< OpenGlRenderer::InstanceId, void > {
    public:
        RemoveInstanceFromRenderMeshLibrary( OpenGlRenderer* _renderer) : renderer_( _renderer ) {
        }
        
        void operator()( OpenGlRenderer::InstanceId const& _id ) {
            renderer_->removeFromRenderMeshLibrary( _id );
        }
        
        
    private:
        OpenGlRenderer* renderer_;
    }; // RemoveInstanceFromRenderMeshLibrary
    
    
    typedef std::vector< OpenGlRenderer::InstanceId > RendererInstanceIdContainer;
    typedef std::map< OpenGlRenderFeeder::InstanceId, RendererInstanceIdContainer > GraphicsPrimitiveIdToRenderInstanceIdsMapping;
    
    
    /**
     * Functor returning @c true if a given render mesh instance id is not 
     * constined inside a specified renderer, @c false otherwise.
     */
    class NotContainedInRenderMeshLibrary : public std::unary_function< OpenGlRenderer::InstanceId, bool > {
    public:
        NotContainedInRenderMeshLibrary( OpenGlRenderer* _renderer ) : renderer_( _renderer ) {
        }
        
        bool operator() ( OpenGlRenderer::InstanceId const& _id ) {
            return ! ( renderer_->inRenderMeshLibrary( _id ) );
        }
        
    private:
        OpenGlRenderer* renderer_;
        
    }; // class NotContainedInRenderMeshLibrary
    
    
    
    
    /**
     * Returns @c true if @a _id is contained in @a _mapping and if all 
     * render mesh library instance ids associated with @a _id are contained in
     * @a _renderer. Returns @c true if @a _id isn't contained in @a mapping and
     * @c false in all other cases.
     */
    bool associatedRenderMeshLibraryInstancesExist( GraphicsPrimitiveIdToRenderInstanceIdsMapping const& _mapping,
                                                    OpenGlRenderFeeder::InstanceId const& _id,
                                                    OpenGlRenderer* _renderer ) {
        
        typedef GraphicsPrimitiveIdToRenderInstanceIdsMapping::const_iterator const_iterator;
        
        const_iterator iter = _mapping.find( _id );
        
        if ( _mapping.end() != iter ) {
            typedef std::vector< OpenGlRenderer::InstanceId >::const_iterator const_vec_iterator;
            const_vec_iterator i = std::find_if( (*iter).second.begin(), (*iter).second.end(), NotContainedInRenderMeshLibrary( _renderer ) );
            
            if ( (*iter).second.end() != i ) {
                return false;
            }
            
        }
       
        
        
        return true;
    }
    
    
    /**
     * Functor that inserts a given OpenGL render mesh into a specified
     * renderer mesh library and returns the assigned render mesh library
     * instance id.
     */
    class InsertIntoRenderMeshLibrary : std::unary_function< SharedPointer< OpenGlRenderMesh >, OpenGlRenderer::InstanceId > {
    public:
        InsertIntoRenderMeshLibrary( OpenGlRenderer* _renderer ) : renderer_( _renderer ) {
            
        }
        
        OpenGlRenderer::InstanceId operator() ( SharedPointer< OpenGlRenderMesh > const& _mesh ) {
            OpenGlRenderer::InstanceId id = 0;
            renderer_->addToRenderMeshLibrary( _mesh, id );
            return id;
        }
        
        
    private:
        OpenGlRenderer* renderer_;
    }; // InsertIntoRenderMeshLibrary
    
    
    class AddUntransformedMeshToRenderer : public std::unary_function< SharedPointer< OpenGlRenderMesh >, void > {
    public:
        AddUntransformedMeshToRenderer( OpenGlRenderer* _renderer ) : renderer_( _renderer ) {
        }
        
        void operator() ( SharedPointer< OpenGlRenderMesh > const& _mesh ) {
            renderer_->addToRender( _mesh );
        }
        
        
    private:
        OpenGlRenderer* renderer_;
    }; // class AddUntransformedMeshToRenderer
    
    
    
    
    class AddTransformedMeshToRenderer : public std::unary_function< SharedPointer< OpenGlRenderMesh >, void > {
    public:
        AddTransformedMeshToRenderer( Matrix const& _transformation, OpenGlRenderer* _renderer ) : transformation_( _transformation), renderer_( _renderer ) {
        }
        
        void operator() ( SharedPointer< OpenGlRenderMesh > const& _mesh ) {
            renderer_->addToRender( transformation_, _mesh );
        }
        
        
    private:
        Matrix transformation_;
        OpenGlRenderer* renderer_;
    }; // class AddTransformedMeshToRenderer
    
    
    
} // anonymous namespace



OpenSteer::Graphics::OpenGlGraphicsPrimitiveToRendererTranslator::~OpenGlGraphicsPrimitiveToRendererTranslator()
{
    // Nothing to do.
}



OpenSteer::Graphics::OpenGlRenderService::~OpenGlRenderService()
{
    // Nothing to do.
}



OpenSteer::SharedPointer< OpenSteer::Graphics::Renderer > 
OpenSteer::Graphics::OpenGlRenderService::createRenderer() const
{
    return SharedPointer<OpenSteer::Graphics::Renderer>( new OpenGlRenderer() );
}



OpenSteer::SharedPointer< OpenSteer::Graphics::RenderFeeder > 
OpenSteer::Graphics::OpenGlRenderService::createRenderFeeder( SharedPointer< Renderer > const& _renderer ) const
{
    SharedPointer< OpenGlRenderer > oglRenderer = dynamic_pointer_cast< OpenGlRenderer >( _renderer );
    
    if ( 0 == oglRenderer.get() ) {
        return SharedPointer< RenderFeeder >();
    }
    
    return SharedPointer< RenderFeeder >( new OpenGlRenderFeeder( oglRenderer, translatorLookup_ ) );
}




void 
OpenSteer::Graphics::OpenGlRenderService::insertTranslator( std::type_info const& _typeInfo, SharedPointer< OpenGlGraphicsPrimitiveToRendererTranslator > const& _translator )
{
    translatorLookup_.insert( _typeInfo, _translator );
}



void 
OpenSteer::Graphics::OpenGlRenderService::removeTranslator( std::type_info const& _typeInfo )
{
    translatorLookup_.remove( _typeInfo );
}



bool 
OpenSteer::Graphics::OpenGlRenderService::containsTranslator(  std::type_info const& _typeInfo ) const
{
    return translatorLookup_.contains( _typeInfo );
}



void 
OpenSteer::Graphics::OpenGlRenderService::clearTranslators()
{
    translatorLookup_.clear();
}




    
 

// @todo Get c const back.
OpenSteer::Graphics::OpenGlRenderFeeder::OpenGlRenderFeeder( SharedPointer< OpenGlRenderer > const& _rendererToFeed,
                                                             GraphicsPrimitivesTranslatorMapper< SharedPointer< OpenGlGraphicsPrimitiveToRendererTranslator > > const & _mapper )
: graphicsPrimitiveIdToRenderInstanceIdsMapping_(), renderer_( _rendererToFeed ), translatorLookup_( _mapper ), nextInstanceId_( 1 )
{
    
}



OpenSteer::Graphics::OpenGlRenderFeeder::~OpenGlRenderFeeder()
{
    // Nothing to do.
}




bool 
OpenSteer::Graphics::OpenGlRenderFeeder::addToGraphicsPrimitiveLibrary( GraphicsPrimitive const& _primitive, InstanceId& _id )
{
    /*
    SharedPointer< OpenGlGraphicsPrimitiveTranslator > primitiveTranslator( translatorLookup_.lookup( typeid( _primitive ) ) );
    
    if ( 0 == primitiveTranslator.get() ) {
        return false;
    }


    SharedPointer< OpenGlRenderMesh > mesh( (*primitiveTranslator)( _primitive ) );
    return renderer_->addToRenderMeshLibrary( mesh, _id );
     */
    
    SharedPointer< OpenGlGraphicsPrimitiveToRendererTranslator > primitiveTranslator( translatorLookup_.lookup( typeid( _primitive ) ) );
    if ( 0 == primitiveTranslator.get() ) {
        std::cerr << "No translator found for graphics primitive." << std::endl;
        return false;
    }

    RendererInstanceIdContainer renderInstances;
    bool const added = primitiveTranslator->addToLibrary( _primitive, *renderer_, renderInstances );
    
    if ( added ) {
        
        // graphicsPrimitiveIdToRenderInstanceIdsMapping_[ nextInstanceId_ ] = renderInstances;
        if ( ! graphicsPrimitiveIdToRenderInstanceIdsMapping_.insert( std::make_pair( nextInstanceId_, renderInstances ) ).second  ) {
            
            std::for_each( renderInstances.begin(), renderInstances.end(), RemoveInstanceFromRenderMeshLibrary( renderer_.get() ) );
            return false;
        }
        _id = nextInstanceId_;
        ++nextInstanceId_;
    }
    
    return added;
    
    
    /* replaced by new translator architecture that just returns the instance ids.
    OpenGlGraphicsPrimitiveTranslator::MeshContainer meshes;
    (*primitiveTranslator)( _primitive, meshes );
    RendererInstanceIdContainer renderInstances( meshes.size() );
    std::transform( meshes.begin(), meshes.end(), renderInstances.begin(), InsertIntoRenderMeshLibrary( renderer_.get() ) );
    
    _id = nextInstanceId_;
    graphicsPrimitiveIdToRenderInstanceIdsMapping_[ _id ] = renderInstances;
    ++nextInstanceId_;
    
    return true;
     */
}



/*
bool 
OpenSteer::Graphics::OpenGlRenderFeeder::addToGraphicsPrimitiveLibrary( Matrix const& _transformation, 
                                                                        GraphicsPrimitive const& _primitive, 
                                                                        InstanceId& _id )
{
    
}
*/


void 
OpenSteer::Graphics::OpenGlRenderFeeder::removeFromGraphicsPrimitiveLibrary( InstanceId const& _id )
{
    // renderer_->removeFromRenderMeshLibrary( _id );
    
    typedef GraphicsPrimitiveIdToRenderInstanceIdsMapping::iterator iterator;
    
    iterator iter = graphicsPrimitiveIdToRenderInstanceIdsMapping_.find( _id );
    
    if ( graphicsPrimitiveIdToRenderInstanceIdsMapping_.end() != iter ) {
        std::for_each( (*iter).second.begin(), (*iter).second.end(), RemoveInstanceFromRenderMeshLibrary( renderer_.get() ) );
        graphicsPrimitiveIdToRenderInstanceIdsMapping_.erase( iter );
    }
    
}


/**
 * @todo Because of the use of a default translator this member function might 
 * be no help anymore...
 */
bool 
OpenSteer::Graphics::OpenGlRenderFeeder::inGraphicsPrimitiveLibrary( InstanceId const& _id ) const
{
    // return renderer_->inRenderMeshLibrary( _id );
    
    // @todo Write an assertion that all render mesh library instances 
    //       associated with @a _id are really contained in @a renderer_.
    assert( associatedRenderMeshLibraryInstancesExist( graphicsPrimitiveIdToRenderInstanceIdsMapping_, _id, renderer_.get() )  );
    
    return ( graphicsPrimitiveIdToRenderInstanceIdsMapping_.find( _id ) != graphicsPrimitiveIdToRenderInstanceIdsMapping_.end() );
    
}



void 
OpenSteer::Graphics::OpenGlRenderFeeder::clearGraphicsPrimitiveLibrary()
{
    renderer_->clearRenderMeshLibrary();
    graphicsPrimitiveIdToRenderInstanceIdsMapping_.clear();
}




void 
OpenSteer::Graphics::OpenGlRenderFeeder::render( Matrix const& _transformation, InstanceId const& _instanceId )
{
    typedef GraphicsPrimitiveIdToRenderInstanceIdsMapping::iterator iterator;    
    iterator iter = graphicsPrimitiveIdToRenderInstanceIdsMapping_.find( _instanceId );
    
    if ( graphicsPrimitiveIdToRenderInstanceIdsMapping_.end() != iter ) {
        std::for_each( (*iter).second.begin(), (*iter).second.end(), AddTransformedInstanceToRender( _transformation, renderer_.get() ) );
    }
}



void 
OpenSteer::Graphics::OpenGlRenderFeeder::render( InstanceId const& _instanceId )
{
    typedef GraphicsPrimitiveIdToRenderInstanceIdsMapping::iterator iterator;
    iterator iter = graphicsPrimitiveIdToRenderInstanceIdsMapping_.find( _instanceId );
    
    if ( graphicsPrimitiveIdToRenderInstanceIdsMapping_.end() != iter ) {
        std::for_each( (*iter).second.begin(), (*iter).second.end(), AddUntransformedInstanceToRender( renderer_.get() ) );
    }  
}



void 
OpenSteer::Graphics::OpenGlRenderFeeder::render( Matrix const& _transformation, GraphicsPrimitive const& _primitive )
{
    SharedPointer< OpenGlGraphicsPrimitiveToRendererTranslator > primitiveTranslator( translatorLookup_.lookup( typeid( _primitive ) ) );
   
    if ( 0 == primitiveTranslator.get() ) {
        std::cerr << "No translator found for graphics primitive." << std::endl;
        return;
    }
    
    primitiveTranslator->translate( _transformation, _primitive, *renderer_ );
    
    // OpenGlGraphicsPrimitiveTranslator::MeshContainer meshes;
    
    // (*primitiveTranslator)( _primitive, meshes );

    // std::for_each( meshes.begin(), meshes.end(), AddTransformedMeshToRenderer( _transformation, renderer_.get() ) );

    // renderer_->addToRender( _transformation, mesh );
}



void 
OpenSteer::Graphics::OpenGlRenderFeeder::render( GraphicsPrimitive const& _primitive )
{
    SharedPointer< OpenGlGraphicsPrimitiveToRendererTranslator > primitiveTranslator( translatorLookup_.lookup( typeid( _primitive ) ) );
    /*  Not needed anymore because of default translator.
    if ( 0 == primitiveTranslator.get() ) {
        std::cerr << "No translator found for graphics primitive." << std::endl;
        return;
    }
    */
    primitiveTranslator->translate( _primitive, *renderer_ );
    
    // SharedPointer< OpenGlRenderMesh > mesh( (*primitiveTranslator)( _primitive ) );
    // renderer_->addToRender( mesh );    
    
    // OpenGlGraphicsPrimitiveTranslator::MeshContainer meshes;
    
    // (*primitiveTranslator)( _primitive, meshes );

    // std::for_each( meshes.begin(), meshes.end(), AddUntransformedMeshToRenderer( renderer_.get() ) );
    
}





// GraphicsPrimitiveToRenderMapper< OpenGlGraphicsPrimitiveTranslator >& translatorLookup_;
// SharedPointer< OpenGlRenderer > renderer_;





