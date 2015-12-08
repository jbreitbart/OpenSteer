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
 * Unit tests for @c randomizer.h.
 *
 * @todo How to test a simple random number generator? Have to read about the mathematical theory
 *       behind randomness (sprectral analysis?).
 */

#include <UnitTest++/UnitTest++.h>

#include "kapaga/randomizer.h"

// Include RAND_MAX, std::size_t
#include <cstdlib>



std::size_t const sequence_count = 10000;


SUITE( randomizer_test ) {
	

	
	TEST( min_and_max )
	{
		using namespace kapaga;
		
		CHECK_EQUAL( int( 0 ), randomizer< int >::min() );
		CHECK_EQUAL( int( RAND_MAX ), randomizer< int >::max() );		
		
		CHECK_EQUAL( float( 0 ), randomizer< float >::min() );
		CHECK_EQUAL( float( 1 ), randomizer< float >::max() );		
		
	}
	
	
	TEST( draw_in_range_int ) 
	{
		using namespace kapaga;
		
		// @todo Merge @c random_number_in_range_float and @c random_number_in_range_int.
		// Random seed.
		random_number_source::source_type seed = random_number_source::default_seed();
		random_number_source source( seed );
		typedef int number_type;
		number_type const lower_bound( 23 );
		number_type const higher_bound( 42 );
		
		typedef std::size_t size_type;
		size_type const test_loop_count = sequence_count;
		for ( size_type i = 0; i < test_loop_count; ++i ) {
			number_type r = randomizer< number_type >::draw( lower_bound, higher_bound, source );
			CHECK( r >= lower_bound );
			CHECK( r <= higher_bound );
		}
		
		
		for ( size_type i = 0; i < test_loop_count; ++i ) {
			number_type r = randomizer< number_type >::draw( source );
			CHECK( r >= randomizer< number_type >::min() );
			CHECK( r <= randomizer< number_type >::max() );
		}
		
	}
	

	
	TEST( draw_in_range_float ) 
	{
		using namespace kapaga;
		
		// @todo Merge @c random_number_in_range_float and @c random_number_in_range_int.
		// Random seed.
		random_number_source::source_type seed = random_number_source::default_seed();
		random_number_source source( seed );
		typedef float number_type;
		number_type const lower_bound( 23 );
		number_type const higher_bound( 42 );
		
		typedef std::size_t size_type;
		size_type const test_loop_count = sequence_count;
		for ( size_type i = 0; i < test_loop_count; ++i ) {
			number_type r = randomizer< number_type >::draw( lower_bound, higher_bound, source );
			CHECK( r >= lower_bound );
			CHECK( r <= higher_bound );
		}
		
		for ( size_type i = 0; i < test_loop_count; ++i ) {
			number_type r = randomizer< number_type >::draw( source );
			CHECK( r >= randomizer< number_type >::min() );
			CHECK( r <= randomizer< number_type >::max() );
		}
		
	}
	

	
	
	
} // SUITE(random_number_source_test)
