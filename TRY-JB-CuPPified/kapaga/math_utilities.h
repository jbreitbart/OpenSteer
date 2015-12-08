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
 * Collection of basic math utilities.
 */
#ifndef KAPAGA_kapaga_math_utilities_H
#define KAPAGA_kapaga_math_utilities_H

#include "kapaga/assert.h"

namespace kapaga {
	
	/**
	 * Returns the number @c pi.
	 */
	template< typename T >
	T pi() 
	{
		// @todo Is this number precise enough for long double types?
		return T(3.14159265358979323846);
	}
	
	 /**
	 * Clamps (limits) the value of @a value_to_clamp to be @a higher_bound at max.
	 *
	 * @return @a value_to_clamp if it is lesser or equal to @a higher_bound, 
	 *         otherwise returns @a higher_bound.
	 */
	template< typename T >
	T clamp_high( T value_to_clamp, T higher_bound )
	{
		return value_to_clamp > higher_bound ? higher_bound : value_to_clamp;
	}

	/**
	 * Clamps (limits) the value of @a value_to_clamp to be @a lower_bound at min.
	 *
	 * @return @a value_to_clamp if it is greater or equal to @a lower_bound, 
	 *         otherwise returns @a lower_bound.
	 */
	template< typename T >
	T clamp_low( T value_to_clamp, T lower_bound )
	{
		return value_to_clamp < lower_bound ? lower_bound : value_to_clamp;
	}

	/**
	 * Clamps (limits) @a value_to_clamp to be @a lower_bound at min and @c higher_bound at max.
	 *
	 * @return @a value_to_clamp if it is greater or equal to @a lower_bound or lesser or equal to
	 *         @a higher_bound. Returns @a lower_bound if @a value_to_clamp is lesser than 
	 *         @a lower_bound. Returns @a higher_bound if @a value_to_clamp is greater than
	 *         @a higher_bound.
	 *
	 * @pre @a lower_bound must be lesser or equal to @a higher_bound, otherwise behavior is 
	 *      undefined.
	 */
	template< typename T >
	T clamp( T value_to_clamp, T lower_bound, T higher_bound )
	{
		KAPAGA_ASSERT( lower_bound <= higher_bound &&
				"Error: lower_bound must be lesser or equal to higher bound otherwise behavior is undefined." );
		if ( value_to_clamp < lower_bound ) {
			value_to_clamp = lower_bound;
		} else if ( value_to_clamp > higher_bound ) {
			value_to_clamp = higher_bound;
		}
		
		return value_to_clamp;
	}
	
	
} // namespace kapaga


#endif // KAPAGA_kapaga_math_utilities_H
