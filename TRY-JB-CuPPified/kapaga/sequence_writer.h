/**
 * Kapaga: Kassel Parallel Games
 *
 * Copyright (c) 2006-2007, Kapaga Development Group
 * All rights reserved.
 *
 * This file is part of the Kapaga project.
 * For conditions of distribution and use, see copyright notice in kapaga_license.txt.
 */

/**
 * @file
 *
 * Helper functions and classes to write a sequence (represented by a begin and end iterator) to
 * and outstream using special delimiters for its beginning and ending and to seperate the
 * elements of the stream.
 *
 * @todo Write unit test that fills a strstream with a sequence and compares the result with what
 *       I expected.
 */

#ifndef KAPAGA_kapaga_sequence_writer_H
#define KAPAGA_kapaga_sequence_writer_H

// @todo Would it be ok to just include the <em>iosfwd</em> header?
// Include std::ostream
//#include <ostream>
#include <iosfwd>

// Include std::iterator_traits
#include <iterator>

// Include std::for_each
#include <algorithm>


// Include PLM_ASSERT
#include "kapaga/assert.h"




namespace kapaga {
    
    /**
	 * Functor to be passed to @c std::for_each to print every iterator inside
     * a given iterator range.
     *
	 * @attention Does not take over memory management for @a _delimiter.
	 *
     * @todo This might not handle stream configurations in the right way...
     */
    template< typename T, typename CharT, class Traits >
	class sequence_writer {
	public:
		sequence_writer( std::basic_ostream< CharT, Traits>& _ostrm, CharT const* _delimiter = ", " ) : delimiter_( _delimiter ), has_been_called_( false ), ostr_( _ostrm ) { 
			// Nothing to do.
		}
		
		
		void operator()( T const& _element ) { 
			if ( has_been_called_ ) {
				ostr_ << delimiter_ << _element;
			} else {
				has_been_called_ = true;
				ostr_ << _element;
			}
		} 
        
	private:
		CharT const* delimiter_;
		bool has_been_called_;
		std::basic_ostream< CharT, Traits>& ostr_;
	}; // class sequence_writer
    
    
    /**
	 * Utility function helping to construct a @c sequence_writer.
	 *
	 * @todo Really useful? Calling the functions always needs the declaration of the first template
	 *       parameter because it can't be deduced from the function arguments.
     */
    template< typename T, typename CharT, class Traits >
        sequence_writer< T, CharT, Traits >
        make_sequence_writer( std::basic_ostream< CharT, Traits>& _ostrm, CharT const* _delimiter = ", "  ) {
            
            return sequence_writer<T, CharT, Traits>( _ostrm, _delimiter );
        }
    
    
    /**
	 * Prints a sequence of values.
     */
    template< typename InputIterator, typename CharT, class Traits >
        void
        write_sequence( InputIterator _first, 
                        InputIterator _last, 
                        std::basic_ostream< CharT, Traits>& _ostrm, 
                        CharT const* _delimiter = ", ",
                        CharT const* _left_delimiter = "( ", 
                        CharT const* _right_delimiter = " )" ) {
            KAPAGA_ASSERT( _last > _first && "Error: _last must be greater or equal than _first." );
            _ostrm << _left_delimiter;
            std::for_each( _first, _last, make_sequence_writer< typename std::iterator_traits< InputIterator >::value_type >( _ostrm, _delimiter ) );
            _ostrm << _right_delimiter;
            
		}
    
    
} // namespace kapaga

#endif // KAPAGA_kapaga_sequence_writer_H
