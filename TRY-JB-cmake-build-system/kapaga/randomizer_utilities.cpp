/**
 * Kapaga: Kassel Parallel Games
 *
 * Copyright (c) 2006-2007, Kapaga Development Group
 * All rights reserved.
 *
 * This file is part of the Kapaga project.
 * For conditions of distribution and use, see copyright notice in kapaga_license.txt.
 */

#include "kapaga/randomizer_utilities.h"

// Include kapaga::randomizer< float >
#include "kapaga/randomizer.h"

// Include KAPAGA_ASSERT
#include "kapaga/assert.h"


float 
kapaga::binomial_randf( random_number_source& source )
{
	// The min and max value of the @c randomizer and the @c random_number_source should be 
	// compile-time testable but then including them would populate the global namespace with
	// symbols from the stdlib header - and I don't want to do this. Back to run-time assertions...
	// KAPAGA_COMPILE_TIME_ASSERT( 0.0f == randomizer< float >::min(), Error_randomizer_must_have_a_min_of_0 );
	// KAPAGA_COMPILE_TIME_ASSERT( 1.0f == randomizer< float >::max(), Error_randomizer_must_have_a_max_of_1 );
	KAPAGA_ASSERT( 0.0f == randomizer< float >::min() && "Error: randomizer must have a min of 0.0f." );
	KAPAGA_ASSERT( 1.0f == randomizer< float >::max() && "Error: randomizer must have a max of 1.0f." );
	
	
	float const rand_one = randomizer< float >::draw( source );
	float const rand_two = randomizer< float >::draw( source );
	
	return rand_one - rand_two;
}

