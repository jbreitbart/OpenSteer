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
 * Unit tests for @c math_utilities.h.
 */

// Include assert
#include <cassert>

#include <UnitTest++/UnitTest++.h>

#include "kapaga/math_utilities.h"


namespace {
	
	/**
	 * Checks if @c clamp_high doesn't clamp @a not_to_clamp and does clamp @a to_clamp and also
	 * tests for negative numbers and corner-cases 
	 * (one greater or one lesser than @a clamp_boundary).
	 *
	 * @pre @c T should be a number type.
	 * @pre @a clamp_boundary must be greater than @c 0, otherwise behavior is undefined.
	 * @pre @a not_clamp must be lesser than @a clamp_boundary, otherwise behavior is undefined.
	 * @pre @a to_clamp must be greater than @a clamp_boundary, otherwise behavior is undefined.
	 */
	template< typename T, typename Details, typename TestResults >
	void check_clamp_high( T clamp_boundary, 
						   T not_to_clamp, 
						   T to_clamp,
						   Details& m_details,
						   TestResults& testResults_ ) {
		assert( T( 0 ) < clamp_boundary && 
				"Error: 0 must be lesser than clamp_boundary, otherwise behavior is undefined.");
		assert( not_to_clamp < clamp_boundary && 
				"Error: not_to_clamp must be lesser than clamp_boundary, otherwise behavior is undefined." );
		assert(  to_clamp > clamp_boundary && 
				"Error: to_clamp must be greater than clamp_boundary, otherwise behavior is undefined." );
		
		using kapaga::clamp_high;
		
		// Checks for the positive clamp_boundary.
		CHECK_EQUAL( T( 0 ) , clamp_high( T( 0 ), clamp_boundary ) );
		CHECK_EQUAL( not_to_clamp, clamp_high( not_to_clamp, clamp_boundary ) );
		CHECK_EQUAL( clamp_boundary, clamp_high( to_clamp, clamp_boundary ) );
		CHECK_EQUAL( clamp_boundary - 1, clamp_high( clamp_boundary - 1, clamp_boundary ) );
		CHECK_EQUAL( clamp_boundary, clamp_high( clamp_boundary + 1, clamp_boundary ) );
		CHECK_EQUAL( clamp_boundary, clamp_high( clamp_boundary, clamp_boundary ) );
		CHECK_EQUAL( -not_to_clamp, clamp_high( -not_to_clamp, clamp_boundary ) );
		CHECK_EQUAL( -to_clamp, clamp_high( -to_clamp, clamp_boundary ) );
		CHECK_EQUAL( -clamp_boundary, clamp_high( -clamp_boundary, clamp_boundary ) );
		CHECK_EQUAL( -clamp_boundary - 1, clamp_high( -clamp_boundary - 1, clamp_boundary ) );
		CHECK_EQUAL( -clamp_boundary + 1, clamp_high( -clamp_boundary + 1, clamp_boundary ) );
		
		// Checks for the negative clamp_boundary.
		T const neg_clamp_boundary = -clamp_boundary;
		T const neg_not_to_clamp = -to_clamp;
		T const neg_to_clamp = -not_to_clamp;
		
		CHECK_EQUAL( neg_clamp_boundary , clamp_high( T( 0 ), neg_clamp_boundary ) );
		CHECK_EQUAL( neg_not_to_clamp, clamp_high( neg_not_to_clamp, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary, clamp_high( neg_to_clamp, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary - 1, clamp_high( neg_clamp_boundary - 1, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary, clamp_high( neg_clamp_boundary + 1, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary, clamp_high( neg_clamp_boundary, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary, clamp_high( -neg_not_to_clamp, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary, clamp_high( -neg_to_clamp, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary, clamp_high( -neg_clamp_boundary, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary, clamp_high( -neg_clamp_boundary - 1, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary, clamp_high( -neg_clamp_boundary + 1, neg_clamp_boundary ) );
	}
	
	
	
	/**
	 * Checks if @c clamp_low doesn't clamp @a not_to_clamp and does clamp @a to_clamp and also
	 * tests for negative numbers and corner-cases 
	 * (one greater or one lesser than @a clamp_boundary).
	 *
	 * @pre @c T should be a number type.
	 * @pre @a clamp_boundary must be greater than @c 0, otherwise behavior is undefined.
	 * @pre @a not_clamp must be greater than @a clamp_boundary, otherwise behavior is undefined.
	 * @pre @a to_clamp must be less than @a clamp_boundary but greater than @c 0, 
     *      otherwise behavior is undefined.
	 */
	template< typename T, typename Details, typename TestResults >
	void check_clamp_low( T clamp_boundary, 
						  T not_to_clamp, 
						  T to_clamp,
						  Details& m_details,
						  TestResults& testResults_ ) {
		
		assert( T( 0 ) < clamp_boundary && 
				"Error: 0 must be lesser than clamp_boundary, otherwise behavior is undefined." );
		assert( not_to_clamp > clamp_boundary && 
				"Error: not_to_clamp must be greater than clamp_boundary, otherwise behavior is undefined." );
		assert( to_clamp < clamp_boundary && 
				"Error: to_clamp must be lesser than clamp_boundary, otherwise behavior is undefined." );
		assert( to_clamp > T( 0 ) && 
				"Error: to_clamp must be greater than 0, otherwise behavior is undefined." );
		
		using kapaga::clamp_low;
		
		CHECK_EQUAL( clamp_boundary, clamp_low( T( 0 ), clamp_boundary ) );
		CHECK_EQUAL( not_to_clamp, clamp_low( not_to_clamp, clamp_boundary ) );
		CHECK_EQUAL( clamp_boundary, clamp_low( to_clamp, clamp_boundary ) );
		CHECK_EQUAL( not_to_clamp - 1, clamp_low( not_to_clamp - 1, clamp_boundary ) );
		CHECK_EQUAL( not_to_clamp + 1, clamp_low( not_to_clamp + 1, clamp_boundary ) );
		CHECK_EQUAL( clamp_boundary, clamp_low( to_clamp - 1, clamp_boundary ) );
		CHECK_EQUAL( clamp_boundary, clamp_low( to_clamp + 1, clamp_boundary ) );
		CHECK_EQUAL( clamp_boundary, clamp_low( clamp_boundary, clamp_boundary ) );
		CHECK_EQUAL( clamp_boundary, clamp_low( clamp_boundary - 1, clamp_boundary ) );
		CHECK_EQUAL( clamp_boundary + 1, clamp_low( clamp_boundary + 1, clamp_boundary ) );
		CHECK_EQUAL( clamp_boundary, clamp_low( -clamp_boundary, clamp_boundary ) );
		CHECK_EQUAL( clamp_boundary, clamp_low( -to_clamp, clamp_boundary ) );
		CHECK_EQUAL( clamp_boundary, clamp_low( -not_to_clamp, clamp_boundary ) );
		
		// Checks for the negative clamp_boundary.
		T const neg_clamp_boundary = -clamp_boundary;
		T const neg_not_to_clamp = -to_clamp;
		T const neg_to_clamp = -not_to_clamp;
		
		CHECK_EQUAL( T( 0 ), clamp_low( T( 0 ), neg_clamp_boundary ) );
		CHECK_EQUAL( neg_not_to_clamp, clamp_low( neg_not_to_clamp, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary, clamp_low( neg_to_clamp, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_not_to_clamp - 1, clamp_low( neg_not_to_clamp - 1, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_not_to_clamp + 1, clamp_low( neg_not_to_clamp + 1, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary, clamp_low( neg_to_clamp - 1, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary, clamp_low( neg_to_clamp + 1, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary, clamp_low( neg_clamp_boundary, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary, clamp_low( neg_clamp_boundary - 1, neg_clamp_boundary ) );
		CHECK_EQUAL( neg_clamp_boundary + 1, clamp_low( neg_clamp_boundary + 1, neg_clamp_boundary ) );
		CHECK_EQUAL( -neg_clamp_boundary, clamp_low( -neg_clamp_boundary, neg_clamp_boundary ) );
		CHECK_EQUAL( -neg_to_clamp, clamp_low( -neg_to_clamp, neg_clamp_boundary ) );
		CHECK_EQUAL( -neg_not_to_clamp, clamp_low( -neg_not_to_clamp, neg_clamp_boundary ) );
	}
	
	
	
	/**
	 * Checks if @c clamp doesn't clamp @a not_to_clamp and does clamp @a low_to_clamp and 
	 * @a high_to_clamp and also tests for negative numbers and corner-cases 
	 * (one greater or one lesser than @a low_clamp_boundary and also @a high_clamp_boundary).
	 *
	 * @pre @c T should be a number type.
	 * @pre @a low_clamp_boundary must be greater than @c 0, otherwise behavior is undefined.
	 * @pre @a high_clamp_boundary must be greater than @c low_clamp_boundary, otherwise beahvior is
	 *      undefined.
	 * @pre @a not_clamp must be greater than @a low_clamp_boundary and lesser than 
	 *      @c high_clamp_boundary, otherwise behavior is undefined.
	 * @pre @a low_to_clamp must be less than @a low_clamp_boundary but greater than @c 0, 
     *      otherwise behavior is undefined.
	 * @pre @a high_to_clamp must be greater than @a high_clamp_boundary, otherwise behavior is
	 *      undefined.
	 */
	template< typename T, typename Details, typename TestResults >
	void check_clamp( T low_clamp_boundary, 
					  T high_clamp_boundary, 
					  T low_to_clamp, 
					  T not_to_clamp, 
					  T high_to_clamp,
					  Details& m_details,
					  TestResults& testResults_ ) {
		
		using kapaga::clamp;
		
		assert( low_clamp_boundary > T( 0 ) && 
				"Error: low_clamp_boundary must be greater than 0, otherwise behavior is undefined." );
		assert( high_clamp_boundary > low_clamp_boundary && 
				"Error: high_clamp_boundary must be greater than low_clamp_boundary, otherwise behavior is undefined." );
		assert( low_clamp_boundary < not_to_clamp && 
				"Error: low_clamp_boundary must be lesser than not_to_clamp, otherwise behavior is undefined." );
		assert( not_to_clamp < high_clamp_boundary && 
				"Error: not_to_clamp must be lesser than high_clamp_boundary, otherwise behavior is undefined." );
		assert( low_to_clamp < low_clamp_boundary && 
				"Error: low_to_clamp mus be lesser than low_clamp_boundary, otherwise behavior is undefined." );
		assert( low_to_clamp > T( 0 ) && 
				"Error: low_to_clamp must be greater than 0, otherwise behavior is undefined." );
		assert( high_to_clamp > high_clamp_boundary && 
				"Error: high_to_clamp must be greater than high_clamp_boundary, otherwise behavior is undefined." );
		
		
		CHECK_EQUAL( low_clamp_boundary, clamp( T( 0 ), low_clamp_boundary, high_clamp_boundary ) );
		
		CHECK_EQUAL( not_to_clamp, clamp( not_to_clamp, low_clamp_boundary, high_clamp_boundary ) );
		CHECK_EQUAL( not_to_clamp + 1, clamp( not_to_clamp + 1, low_clamp_boundary, high_clamp_boundary ) );
		CHECK_EQUAL( not_to_clamp - 1, clamp( not_to_clamp - 1, low_clamp_boundary, high_clamp_boundary ) );
					 
		CHECK_EQUAL( low_clamp_boundary, clamp( low_to_clamp, low_clamp_boundary, high_clamp_boundary ) );	
		CHECK_EQUAL( low_clamp_boundary, clamp( low_to_clamp + 1, low_clamp_boundary, high_clamp_boundary ) );
		CHECK_EQUAL( low_clamp_boundary, clamp( low_to_clamp - 1, low_clamp_boundary, high_clamp_boundary ) );	
		CHECK_EQUAL( high_clamp_boundary, clamp( high_to_clamp, low_clamp_boundary, high_clamp_boundary ) );
		CHECK_EQUAL( high_clamp_boundary, clamp( high_to_clamp + 1, low_clamp_boundary, high_clamp_boundary ) );
		CHECK_EQUAL( high_clamp_boundary, clamp( high_to_clamp - 1, low_clamp_boundary, high_clamp_boundary ) );
		
		CHECK_EQUAL( low_clamp_boundary, clamp( -low_to_clamp, low_clamp_boundary, high_clamp_boundary ) );
		CHECK_EQUAL( low_clamp_boundary, clamp( -not_to_clamp, low_clamp_boundary, high_clamp_boundary ) );
		CHECK_EQUAL( low_clamp_boundary, clamp( -high_to_clamp, low_clamp_boundary, high_clamp_boundary ) );
		
		CHECK_EQUAL( low_clamp_boundary, clamp( low_clamp_boundary , low_clamp_boundary, high_clamp_boundary ) );
		CHECK_EQUAL( low_clamp_boundary, clamp( low_clamp_boundary - 1 , low_clamp_boundary, high_clamp_boundary ) );
		CHECK_EQUAL( low_clamp_boundary + 1, clamp( low_clamp_boundary + 1 , low_clamp_boundary, high_clamp_boundary ) );
		
		CHECK_EQUAL( high_clamp_boundary, clamp( high_clamp_boundary , low_clamp_boundary, high_clamp_boundary ) );
		CHECK_EQUAL( high_clamp_boundary - 1, clamp( high_clamp_boundary - 1, low_clamp_boundary, high_clamp_boundary ) );
		CHECK_EQUAL( high_clamp_boundary, clamp( high_clamp_boundary + 1, low_clamp_boundary, high_clamp_boundary ) );
		
		
		
		
		T const neg_low_clamp_boundary = -high_clamp_boundary;
		T const neg_high_clamp_boundary = -low_clamp_boundary;
		T const neg_not_to_clamp = -not_to_clamp; 
		T const neg_low_to_clamp = -high_to_clamp;
		T const neg_high_to_clamp = -low_to_clamp;
		
		CHECK_EQUAL( neg_high_clamp_boundary, clamp( T( 0 ), neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		
		CHECK_EQUAL( neg_not_to_clamp, clamp( neg_not_to_clamp, neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		CHECK_EQUAL( neg_not_to_clamp + 1, clamp( neg_not_to_clamp + 1, neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		CHECK_EQUAL( neg_not_to_clamp - 1, clamp( neg_not_to_clamp - 1, neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		
		CHECK_EQUAL( neg_low_clamp_boundary, clamp( neg_low_to_clamp, neg_low_clamp_boundary, neg_high_clamp_boundary ) );	
		CHECK_EQUAL( neg_low_clamp_boundary, clamp( neg_low_to_clamp + 1, neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		CHECK_EQUAL( neg_low_clamp_boundary, clamp( neg_low_to_clamp - 1, neg_low_clamp_boundary, neg_high_clamp_boundary ) );	
		CHECK_EQUAL( neg_high_clamp_boundary, clamp( neg_high_to_clamp, neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		CHECK_EQUAL( neg_high_clamp_boundary, clamp( neg_high_to_clamp + 1, neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		CHECK_EQUAL( neg_high_clamp_boundary, clamp( neg_high_to_clamp - 1, neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		
		CHECK_EQUAL( neg_high_clamp_boundary, clamp( -neg_low_to_clamp, neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		CHECK_EQUAL( neg_high_clamp_boundary, clamp( -neg_not_to_clamp, neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		CHECK_EQUAL( neg_high_clamp_boundary, clamp( -neg_high_to_clamp, neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		
		CHECK_EQUAL( neg_low_clamp_boundary, clamp( neg_low_clamp_boundary , neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		CHECK_EQUAL( neg_low_clamp_boundary, clamp( neg_low_clamp_boundary - 1 , neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		CHECK_EQUAL( neg_low_clamp_boundary + 1, clamp( neg_low_clamp_boundary + 1 , neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		
		CHECK_EQUAL( neg_high_clamp_boundary, clamp( neg_high_clamp_boundary , neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		CHECK_EQUAL( neg_high_clamp_boundary - 1, clamp( neg_high_clamp_boundary - 1, neg_low_clamp_boundary, neg_high_clamp_boundary ) );
		CHECK_EQUAL( neg_high_clamp_boundary, clamp( neg_high_clamp_boundary + 1, neg_low_clamp_boundary, neg_high_clamp_boundary ) );
	}
	
} // anonymous namespace


SUITE(math_utilities_test)
{

	TEST(clamp_high)
	{
		check_clamp_high( 42, 4, 123, m_details, testResults_ );
		check_clamp_high( 42.0f, 4.0f, 123.0f, m_details, testResults_ );
		check_clamp_high( 42.0, 4.0, 123.0, m_details, testResults_ );
	}
	
	TEST(clamp_low)
	{
		check_clamp_low( 42, 123, 4, m_details, testResults_ );
		check_clamp_low( 42.0f, 123.0f, 4.0f, m_details, testResults_ );
		check_clamp_low( 42.0, 123.0, 4.0, m_details, testResults_ );
	}
	
	TEST(clamp)
	{
		check_clamp( 23, 42, 4, 32, 123, m_details, testResults_ );
		check_clamp( 23.0f, 42.0f, 4.0f, 32.0f, 123.0f, m_details, testResults_ );
		check_clamp( 23.0, 42.0, 4.0, 32.0, 123.0, m_details, testResults_ );
	}
	
} // SUITE(math_utilities_test)
