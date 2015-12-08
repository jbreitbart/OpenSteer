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
 * Storage for thread specific data (currently just for OpenMP) and access functions that map a
 * thread to a container slot/element.
 *
 * @todo Replace the whole container just by a templated function that takes a sequential container 
 *       via a template parameter and returns the slot whoes index is associated with a thread id.
 *       Later on another template parameter might set a policy how to associate an index into the
 *       container with a thread id.
 * @todo Provide separate headers for @c thread_storage and @c thread_slot_count and @c thread_slot.
 */

#ifndef KAPAGA_kapaga_thread_storage_H
#define KAPAGA_kapaga_thread_storage_H


// Include std::vector
#include <vector>

// Include KAPAGA_ASSERT
#include "kapaga/assert.h"

// Include omp_get_max_threads, omp_get_thread_num
#include "kapaga/omp_header_wrapper.h"

// Include kapaga::size_t
#include "kapaga/standard_types.h"


namespace kapaga {
	
	/**
	 * Storage for OpenMP thread specific data.
	 *
	 * Consists of slots. Each slot is associated to a (possibly) running thread. Inside an OpenMP 
	 * parallel region the @c slot member function admits the calling OpenMP thread access to
	 * its associated slot.
	 * 
	 * Thread storage is of interest if the stored values should be collected in a non-parallel
	 * region after the parallel region has been left.
	 *
	 * @ATTENTION Other than that no thread safety is guaranteed!
	 *
	 * @todo Detect if called from a nested parallel region - which shouldn't happen as OpenMP
	 *       doesn't provide thread ids that allow to differentiate between different levels of
	 *       nested parallelism.
	 *
	 * @todo Write a unit test for this! (Well, normally a unit test should be written before even
	 *       the interface...).
	 * @todo Detect if thread storage doesn't contain pointers and then organize memory 
	 *       (for example by allocating memory and store pointers to this memory in the base array,
	 * adding a new level of indirection. If @c slot is called it only reads where the 
	 * pointer to the real memory points to. Only this pointed to memory is manipulated.
	 */
	template< typename SlotType >
	class thread_storage {
	public:
		
		typedef typename std::vector< SlotType >::reference reference;
		typedef typename std::vector< SlotType >::const_reference const_reference;
		typedef typename std::vector< SlotType >::size_type size_type;
		typedef typename std::vector< SlotType >::difference_type difference_type;
		typedef typename std::vector< SlotType >::value_type value_type;
		
		thread_storage();
		explicit thread_storage( size_type _size, SlotType const& value = SlotType() );
		thread_storage( thread_storage const& other );
		
		thread_storage& operator=( thread_storage other );
		void swap( thread_storage& other );
		
		/**
		 * Returns a reference to the slot value associated to the OpenMP thread calling @c slot.
		 * 
		 * @ATTENTION Don't call from inside a nested parallel region. This might lead to race
		 *            conditions if concurrently a non-nested parallel region calls it.
		 */
		reference slot();
		const_reference slot() const;
		
		/**
		 * Access the specified slot.
		 *
		 * @param index of the slot to access. Must be inside the range @c 0 to @c size().
		 */
		reference accessSlotSequentially( size_type index );
		const_reference accessSlotSequentially( size_type index ) const;
		
		bool empty() const;
		size_type size() const;
		
		/**
		 * Returns the default slot count a @c thread_storage instance will allocate if the standard
		 * constructor is called.
		 */
		static size_type default_size();
		
		
		void resize( size_type newSize, SlotType const& newValue = SlotType() );
		void clear();
		
	private:
		// @todo Check if calling @c operator[] is really thread-safe (in the way used here). 
	    //       What does "returns a lvalue" in the standard mean? Could a proxy be returned? A 
		//       proxy that changes stuff in the vector when called and therefore making this 
	    //       operator not thread-safe?
		std::vector< SlotType > slots_;
	}; // class thread_storage
	
	

	
	
	/**
	 * @return the max number of threads and therefore the number of slots a container should
	 *         at least have to be used by @c thread_slot.
	 *
	 * @todo Put the <em>max</em> and the <em>needed</em> at the end of the name?
	 */
	kapaga::size_t max_needed_thread_slot_count();
	
	
	/**
	 * @return the current number of threads and therefore the number of slots a container should
	 *         have in the current context to be used by @c thread_slot. @c 0 is returned if called
	 *         from a sequential part of the program.
	 *
	 * @todo Put the <em>needed</em> at the end of the name?
	 */
	kapaga::size_t needed_thread_slot_count();
	
	
	/**
	 * @return reference to the container slot (element) assocaited with the callers thread.
	 */
	template< typename Container >
		typename Container::reference
		thread_slot( Container& c );
	
	/**
	 * @return reference to the array slot/element assocaited with the callers thread.
	 *
	 * @pre @a c must have at least @c thread_slot_count elements, otherwise behavior is undefined.
	 */
	template< typename T >
		T&
		thread_slot( T* c );
	
	/**
	 * Calls @c slot on @a ts.
	 */
	template< typename T >
		typename thread_storage< T >::reference
		thread_slot( thread_storage< T >& ts );
	
	
	template< typename Container >
		typename Container::const_reference
		thread_slot( Container const& c );
	
	template< typename T >
		T const&
		thread_slot( T const* c );
	
	template< typename T >
		typename thread_storage< T >::const_reference
		thread_slot( thread_storage< T > const& ts );	
	
	
	
	
	
	
} // namespace kapaga


// Template class defintion.

template< typename SlotType >
kapaga::thread_storage< SlotType >::thread_storage()
: slots_( default_size(), SlotType() )
{
	// Nothing to do.
}


template< typename SlotType >
kapaga::thread_storage< SlotType >::thread_storage( size_type _size, SlotType const& value /* = SlotType() */ )
: slots_( _size, value )
{
	// Nothing to do.
}



template< typename SlotType >
kapaga::thread_storage< SlotType >::thread_storage( thread_storage const& other )
: slots_( other.slots_ )
{
	// Nothing to do.
}

template< typename SlotType >
kapaga::thread_storage< SlotType >& 
kapaga::thread_storage< SlotType >::operator=( thread_storage other )
{
	swap( other );
	return *this;
}


template< typename SlotType >
void 
kapaga::thread_storage< SlotType >::swap( thread_storage& other )
{
	slots_.swap( other.slots );
}


template< typename SlotType >
typename kapaga::thread_storage< SlotType >::reference 
kapaga::thread_storage< SlotType >::slot()
{	
	//KAPAGA_ASSERT( ( size() > static_cast< size_type >( omp_get_thread_num() ) ) && 
	//		"Error: Not enough slots for thread count." );
	//return 	slots_[ static_cast< size_type >( omp_get_thread_num() ) ];

	return thread_slot( slots_ );
}


template< typename SlotType >
typename kapaga::thread_storage< SlotType >::const_reference 
kapaga::thread_storage< SlotType >::slot() const
{
	// @todo Is this an ugly hack or something good because it prevents code duplication?
	return const_cast< thread_storage& >( *this ).slot();
}



template< typename SlotType >
typename kapaga::thread_storage< SlotType >::reference 
kapaga::thread_storage< SlotType >::accessSlotSequentially( size_type index )
{
	KAPAGA_ASSERT( ( index < size() ) && "Error: Index out of range." );
	return slots_[ index ];
}



template< typename SlotType >
typename kapaga::thread_storage< SlotType >::const_reference 
kapaga::thread_storage< SlotType >::accessSlotSequentially( size_type index ) const
{
	// @todo Is this an ugly hack or something good because it prevents code duplication?
	return const_cast< thread_storage< SlotType >& >( *this ).accessSlotSequentially();
}


template< typename SlotType >
bool 
kapaga::thread_storage< SlotType >::empty() const
{
	return slots_.empty();
}

template< typename SlotType >
typename kapaga::thread_storage< SlotType >::size_type 
kapaga::thread_storage< SlotType >::size() const
{
	return slots_.size();
}



template< typename SlotType >
typename kapaga::thread_storage< SlotType >::size_type 
kapaga::thread_storage< SlotType >::default_size()
{
	return max_needed_thread_slot_count();
}


template< typename SlotType >
void 
kapaga::thread_storage< SlotType >::resize( size_type newSize, SlotType const& newValue /* = SlotType() */ )
{
	slots_.resize( newSize, newValue );
}


template< typename SlotType >
void 
kapaga::thread_storage< SlotType >::clear()
{
	slots_.clear();
}







template< typename Container >
typename Container::reference
kapaga::thread_slot( Container& c )
{	
	typedef typename Container::size_type size_type;
	
	KAPAGA_ASSERT( ( c.size() > static_cast< size_type >( omp_get_thread_num() ) ) && 
				   "Error: Not enough slots for thread count." );
	
	return 	c[ static_cast< size_type >( omp_get_thread_num() ) ];	
}



template< typename T >
T&
kapaga::thread_slot( T* c)
{
	return 	c[ static_cast< kapaga::size_t >( omp_get_thread_num() ) ];		
}



template< typename T >
typename kapaga::thread_storage< T >::reference
kapaga::thread_slot( thread_storage< T >& ts )
{
	return ts.slot();
}



template< typename Container >
typename Container::const_reference
kapaga::thread_slot( Container const& c )
{
	return thread_slot( const_cast< Container& >( c ) );
}


template< typename T >
T const&
kapaga::thread_slot( T const* c )
{
	return thread_slot( const_cast< T* >( c ) );
}



template< typename T >
typename kapaga::thread_storage< T >::const_reference
kapaga::thread_slot( thread_storage< T > const& ts )
{
	return ts.slot();
}





#endif // KAPAGA_kapaga_thread_storage_H
