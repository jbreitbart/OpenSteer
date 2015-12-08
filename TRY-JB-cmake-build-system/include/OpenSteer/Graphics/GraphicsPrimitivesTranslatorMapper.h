#ifndef OPENSTEER_GRAPHICS_GRAPHICSPRIMITIVESTRANSLATORMAPPER_H
#define OPENSTEER_GRAPHICS_GRAPHICSPRIMITIVESTRANSLATORMAPPER_H

// Include std::map
#include <map>

// Include std::type_info
#include <typeinfo>


// Include OpenSteer::Matrix
#include "OpenSteer/Matrix.h"

// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"



namespace OpenSteer {

    namespace Graphics {
    
    
        /**
         * Registers translators that translate graphics primitives into render
         * primitives and based on these translators translates graphics
         * primitives.
         *
         * @todo Don't use std::type_info directly but encapuslate it into an
         *       own type info object to lessen dependency on the standard lib
         *       in the public interface.
         *
         * @todo Need to adapt the translator mapper and the render feeder to 
         *       the new possibilities to just add null translators 
         *       (using the translator proxy) if an unknown primitive is tried 
         *       to be rendered. 
         */
        template< class Translator >
        class GraphicsPrimitivesTranslatorMapper {
        public:
            GraphicsPrimitivesTranslatorMapper();
            GraphicsPrimitivesTranslatorMapper( GraphicsPrimitivesTranslatorMapper const& other );
            ~GraphicsPrimitivesTranslatorMapper();
            GraphicsPrimitivesTranslatorMapper& operator=( GraphicsPrimitivesTranslatorMapper other );
            
            void swap( GraphicsPrimitivesTranslatorMapper& other );
            
            
            void insert( std::type_info const& _typeInfo, Translator const& _translator );
            void remove( std::type_info const& _typeInfo );
            bool contains( std::type_info const& _typeInfo ) const;
            void clear();
            
            // Translator& translate( GraphicsPrimitive const& _graphicsPrimitive );
            // Translator& translate( Matrix const& _transformation, GraphicsPrimitive const& _graphicsPrimitive );
            
            /**
             * Returns the translator registerd for the type of 
             * @a _graphicsPrimitive. If no translator is registered a default
             *  translator is returned.
             */
            Translator lookup( std::type_info const& _typeInfo ) const;
            
            
            /**
             * Returns the translator registered for the given type of 
             * @c _graphicsPrimitive. If such a translator hasn't been added
             * a default one is added and returned.
             *
             * @attention Only use this functionality if the used translator
             *            classes allow for safe usage of a default translator.
             */
            // Translator lookup_fast( std::type_info const& _typeInfo );
            
        private:
            typedef std::map< char const*, Translator > LookupContainer; 
            LookupContainer translatorLookup_;   
            
        }; // class GraphicsPrimitivesTranslatorMapper



    } //  namespace Graphics

} // namespace OpenSteer


// Template class implementation.


template< class Translator >
OpenSteer::Graphics::GraphicsPrimitivesTranslatorMapper< Translator >::GraphicsPrimitivesTranslatorMapper() 
: translatorLookup_()
{
    // Nothing to do.
}



template< class Translator >
OpenSteer::Graphics::GraphicsPrimitivesTranslatorMapper< Translator >::GraphicsPrimitivesTranslatorMapper( GraphicsPrimitivesTranslatorMapper const& other ) 
: translatorLookup_( other )
{
    // Nothing to do.
}



template< class Translator >
OpenSteer::Graphics::GraphicsPrimitivesTranslatorMapper< Translator >::~GraphicsPrimitivesTranslatorMapper() 
{
    // Nothing to do.
}



template< class Translator >
OpenSteer::Graphics::GraphicsPrimitivesTranslatorMapper< Translator >& 
OpenSteer::Graphics::GraphicsPrimitivesTranslatorMapper< Translator >::operator=( GraphicsPrimitivesTranslatorMapper other ) 
{
    swap( other );
    return *this;
}



template< class Translator >
void 
OpenSteer::Graphics::GraphicsPrimitivesTranslatorMapper< Translator >::swap( GraphicsPrimitivesTranslatorMapper& other ) 
{
    translatorLookup_.swap( other.translatorLookup_ );
}




template< class Translator >
void 
OpenSteer::Graphics::GraphicsPrimitivesTranslatorMapper< Translator >::insert( std::type_info const& _typeInfo, Translator const& _translator ) 
{
    // Isn't needed anymore because of the shared pointer.
    // Translator** translator =  &( translatorLookup_[ _typeInfo.name() ] );
    // delete *translator;
    // *translator = _translator;
    
    translatorLookup_[ _typeInfo.name() ] = _translator;
}



template< class Translator >
void 
OpenSteer::Graphics::GraphicsPrimitivesTranslatorMapper< Translator >::remove( std::type_info const& _typeInfo ) 
{
    // Unnecessary because of the shared pointer.
    // typedef LookupContainer::iterator iterator;
    // iterator iter = translatorLookup_.find( _typeinfo.name() );
    // if ( iter != translatorLookup_.end() ) {
    //    delete (*iter).second;
    //    translatorLookup_.erase( iter );
    //}
    
    translatorLookup_.erase( _typeInfo.name() );

}



template< class Translator >
bool 
OpenSteer::Graphics::GraphicsPrimitivesTranslatorMapper< Translator >::contains( std::type_info const& _typeInfo ) const 
{
    return translatorLookup_.end() != translatorLookup_.find( _typeInfo.name() );
}



template< class Translator >
void 
OpenSteer::Graphics::GraphicsPrimitivesTranslatorMapper< Translator >::clear() 
{
    translatorLookup_.clear();    
}



template< class Translator >
Translator 
OpenSteer::Graphics::GraphicsPrimitivesTranslatorMapper< Translator >::lookup( std::type_info const& _typeInfo ) const
{
    typedef typename LookupContainer::const_iterator const_iterator;
    const_iterator iter = translatorLookup_.find( _typeInfo.name() );
    if ( translatorLookup_.end() == iter ) {
        std::cerr << "Unknown graphics primitive " << _typeInfo.name() << ". No translation mapping possible." << std::endl;
        return Translator();
    }   
    
    return (*iter).second;
}

/*
template< class Translator >
Translator 
OpenSteer::Graphics::GraphicsPrimitivesTranslatorMapper< Translator >::lookup_fast( std::type_info const& _typeInfo )
{
    return translatorLookup_[ _typeInfo.name() ];
}
*/



#endif // OPENSTEER_GRAPHICS_GRAPHICSPRIMITIVESTRANSLATORMAPPER_H
