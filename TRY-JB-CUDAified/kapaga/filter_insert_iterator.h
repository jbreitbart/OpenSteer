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
 * Output iterator adapter that holds an output iterator and filters every value to insert 
 * through a call to its predicate 
 * 
 */

#ifndef KAPAGA_kapaga_filter_insert_iterator_h
#define KAPAGA_kapaga_filter_insert_iterator_h

// Include std::iterator, std::output_iterator_tag
#include <iterator>

namespace kapaga {
	
	
	/**
	 * Output iterator adapter that holds an output iterator and filters every value to insert 
	 * through a call to its predicate of type @c Predicate.
	 *
	 * For information about output iterators see Ray Lischner, C++ in a Nutshell, O'Reilly 2003, 
	 * pp. 535--553.
	 *
	 * @c Predicate must return a @c bool and must be a pure function 
	 * (see Scott Meyer, Effective STL, Addison-Wesley 2001, pp. 166--169).
	 */
	template< typename OutputIterator, typename Predicate >
	class filter_insert_iterator 
	: public std::iterator< std::output_iterator_tag, void, void, void, void >
	{
	public:
		typedef typename OutputIterator::container_type container_type;
		
		filter_insert_iterator( OutputIterator iter,
								Predicate predicate )
		: predicate_( predicate ), output_iterator_( iter )
		{
			// Nothing to do.
		}
		
		filter_insert_iterator& operator=( typename container_type::const_reference value )
		{
			if ( predicate_( value ) ) {
				output_iterator_ = value;
			}
			return *this;
		}
		
		filter_insert_iterator& operator*() 
		{
			return *this;
		}
		
		filter_insert_iterator& operator++() 
		{
			++output_iterator_;
			return *this;
		}
		
		
		filter_insert_iterator& operator++( int i )
		{
			output_iterator_++;
			return *this;
		}
		
		
		
	private:
		Predicate const predicate_;
		OutputIterator output_iterator_;
	}; // class filter_insert_iterator
	
	
	/**
	 * Constructs a @c filter_insert_iterator that filters assigned values through a @c Predicate 
	 * and only assigns them to the contained @c OutputIterator if the predicate returns @c true for 
	 * the value.
	 */
	template< typename OutputIterator, typename Predicate >
		filter_insert_iterator< OutputIterator, Predicate >
		filter_inserter( OutputIterator iter,
						 Predicate predicate ) {
			filter_insert_iterator< OutputIterator, Predicate> iterator( iter, predicate );
			return iterator;
		}
	
	
} // namespace kapaga



#endif // KAPAGA_kapaga_filter_insert_iterator_h
