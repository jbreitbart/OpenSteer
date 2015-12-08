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
 * Tests the sequential access functionality of the ThreadStorage class and the parallel access to 
 * the @c slot member function.
 */

#include <UnitTest++/UnitTest++.h>

// Include omp_get_max_threads, omp_thread_num
#include "kapaga/omp_header_wrapper.h"


// Include kapaga::size_t
#include "kapaga/standard_types.h"

// Include OpenSteer::ThreadStorage
#include "kapaga/thread_storage.h"



SUITE( thread_storage_test )
{

	// Construct ThreadStorage without the use of threads.
	TEST(default_constructor_test)
	{
		typedef kapaga::thread_storage< float >::size_type size_type;
		kapaga::thread_storage<float> ts;
		// If compiled without OpenMP there should be at maximum one thread and therefore the automatic
		// size of a @c ThreadStorage instance should be @c 1.
		CHECK_EQUAL(ts.size(), static_cast< size_type >( omp_get_max_threads() ) );
		CHECK( !ts.empty() );
	}

	/* Removed because I cahnged the constructors.
	// Constucts ThreadStorage with an init value.
	TEST(construct_with_init_value)
	{
		typedef float ts_value_type;
		ts_value_type const init_value = ts_value_type( 42 );
		OpenSteer::ThreadStorage<ts_value_type> ts( init_value );
		CHECK_EQUAL( ts.size(), omp_get_max_threads() );
		CHECK( !ts.empty() );
		
		// Check that all slots (@c 1 in a sequential app or @c omp_get_max_threads in a parallel
		// app) have the set init value.
		for ( OpenSteer::ThreadStorage< ts_value_type >::size_type i = 0; i < ts.size(); ++i ) {
			CHECK_EQUAL( ts.accessSlotSequentially( i ), init_value );
		}
	}
	*/

	// Construct ThreadStorage with three slots.
	TEST(construct_three_slots)
	{
		kapaga::thread_storage<float>::size_type const slot_count = 3;
		kapaga::thread_storage<float> ts( slot_count );
		CHECK_EQUAL( ts.size(), slot_count );
		CHECK( !ts.empty() );
	}

	// Construct ThreadStorage with three slots and an init value.
	TEST( construct_three_slots_with_init_value )
	{
		typedef float ts_value_type;
		kapaga::thread_storage< ts_value_type >::size_type const slot_count = 3;
		ts_value_type const init_value = ts_value_type( 42 );
		kapaga::thread_storage< ts_value_type > ts( slot_count, init_value );
		
		CHECK_EQUAL( ts.size(), slot_count );
		CHECK( !ts.empty() );
		
		CHECK_EQUAL( ts.accessSlotSequentially( 0 ), init_value );
		CHECK_EQUAL( ts.accessSlotSequentially( 1 ), init_value );
		CHECK_EQUAL( ts.accessSlotSequentially( 2 ), init_value );
	}



	// Access and set a three slot ThreadStorage sequentially.
	TEST(access_and_modify_slots)
	{
		typedef float ts_value_type;
		kapaga::thread_storage< ts_value_type >::size_type const slot_count = 3;
		kapaga::thread_storage< ts_value_type > ts( slot_count );
		
		ts_value_type const val_0 = ts_value_type( 100 );
		ts_value_type const val_1 = ts_value_type( 101 );
		ts_value_type const val_2 = ts_value_type( 102 );
		
		ts.accessSlotSequentially( 0 ) = val_0;
		ts.accessSlotSequentially( 1 ) = val_1;
		ts.accessSlotSequentially( 2 ) = val_2;
		
		CHECK_EQUAL( ts.accessSlotSequentially( 0 ), val_0 );
		CHECK_EQUAL( ts.accessSlotSequentially( 1 ), val_1 );
		CHECK_EQUAL( ts.accessSlotSequentially( 2 ), val_2 );
		
		
	}

	// Access ThreadStorage via @c slot.
	TEST(slot_access)
	{
		typedef float ts_value_type;
		typedef kapaga::thread_storage< ts_value_type > ts_type;
		typedef ts_type::size_type size_type;
		
		ts_type ts;
		
		// Fill @c ts slots with slot-specfic values.
		for ( size_type i = 0; i < static_cast< size_type >( omp_get_max_threads() ); ++i ) {
			ts.accessSlotSequentially( i ) = ts_value_type( 100 + i );
		}
				
		// Test that every thread reads the right slot value and can change it, too.
		#pragma omp parallel default(shared)
		{
			CHECK_EQUAL( ts.accessSlotSequentially( omp_get_thread_num() ), ts.slot() );
			
			ts.slot() = ts_value_type( 200 + omp_get_thread_num() );
			CHECK_EQUAL( ts.accessSlotSequentially( omp_get_thread_num() ), ts.slot() );
			
		} // #pragma omp parallel 
		
		// Control slot values afterwards.
		for ( size_type i = 0; i < static_cast< size_type >( omp_get_max_threads() ); ++i ) {
			CHECK_EQUAL( ts_value_type( 200 + i ), ts.accessSlotSequentially( i ) );
		}	
		
	}

	
	TEST( thread_slot_count )
	{
		// Check that @c thread_slot_count returns the current number of threads.
		
		using namespace kapaga;
		
		CHECK_EQUAL( static_cast< kapaga::size_t >( omp_get_max_threads() ), needed_thread_slot_count() );
		
		CHECK_EQUAL( static_cast< kapaga::size_t >( omp_get_max_threads() ), max_needed_thread_slot_count() );

		
		
		#pragma omp parallel default( shared )
		{
			CHECK_EQUAL( static_cast< kapaga::size_t >( omp_get_num_threads() ), needed_thread_slot_count() );
			
			CHECK_EQUAL( static_cast< kapaga::size_t >( omp_get_max_threads() ), max_needed_thread_slot_count() );
		}
	}
	
	TEST( thread_slot_for_vector )
	{
		// Check that @c thread_slot accesses the container index/slot that is associated with the
		// current thread so no locking is needed if all threads just use @c thread_slot.
		
		using namespace kapaga;
		
		typedef std::vector< int > container_type;
		container_type container( needed_thread_slot_count(), 100 );
		
		for ( int i = 0; i < omp_get_max_threads(); ++i ) {
			container[ i ] += i;
		}
		
		#pragma omp parallel default( shared )
		{
			CHECK_EQUAL( omp_get_thread_num() + 100, thread_slot( container ) );
		}
		
		
		#pragma omp parallel default( shared )
		{
			thread_slot( container ) += 100;
		}
		
		
		for ( int i = 0; i < static_cast< int >( container.size() ); ++i ) {
			CHECK_EQUAL( 200 + i, container[ static_cast< container_type::size_type >( i ) ] );
		}
		
	}
	
	
	TEST( thread_slot_for_array )
	{
		// Check that @c thread_slot accesses the container index/slot that is associated with the
		// current thread so no locking is needed if all threads just use @c thread_slot.
		
		using namespace kapaga;
		
		typedef std::vector< int > container_type;
		container_type container( needed_thread_slot_count(), 100 );
		
		for ( int i = 0; i < omp_get_max_threads(); ++i ) {
			container[ i ] += i;
		}
		
		#pragma omp parallel default( shared )
		{
			CHECK_EQUAL( omp_get_thread_num() + 100, thread_slot( &container[ 0 ] ) );
		}
		
		
		#pragma omp parallel default( shared )
		{
			thread_slot( &container[ 0 ] ) += 100;
		}
		
		
		for ( int i = 0; i < static_cast< int >( container.size() ); ++i ) {
			CHECK_EQUAL( 200 + i, container[ static_cast< container_type::size_type >( i ) ] );
		}
		
	}
	

	// Resize thread_storage.


	// Clear thread_storage.



} // SUITE(thread_storage_test)

