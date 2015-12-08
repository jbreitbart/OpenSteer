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
 * @file wall_clock_time.h
 *
 * Posix specific way to measure wall clock time. Just a thin wrapper for @c gettimeofday.
 *
 * @todo Add overflow tests to the @c convert_time_to functions or offer stand alone overflow test
 *       functions.
 */

//Header guard macro format: KAPAGA_namespace_file_name_H
#ifndef KAPAGA_kapaga_wall_clock_time_H
#define KAPAGA_kapaga_wall_clock_time_H


/* Include gettimeofday, timeval */
#include <sys/time.h>

namespace kapaga {
	
	/**
	 * Opaque type returned by @c wall_clock_time_now that encodes wall clock time or a time 
	 * duration.
	 */
	typedef timeval wall_clock_time_t;
	
	/**
	 * Type that is the natural type when converting wall clock time to a specific time unit.
	 */
	typedef long wall_clock_time_unit_t;
	
	
	extern wall_clock_time_unit_t const wall_clock_time_unit_factor_s_to_ms;
	extern wall_clock_time_unit_t const wall_clock_time_unit_factor_ms_to_us;	
	extern wall_clock_time_unit_t const wall_clock_time_unit_factor_s_to_us;
	
	extern wall_clock_time_unit_t const wall_clock_time_unit_factor_us_to_ms;
	extern wall_clock_time_unit_t const wall_clock_time_unit_factor_ms_to_s;	
	extern wall_clock_time_unit_t const wall_clock_time_unit_factor_us_to_s;
	

	
	/**
	 * Returns the current wall clock time in an opaque type.
	 * Use one of the @c convert_time_to functions to convert it into a specific unit of time.
	 *
	 * Don't use this time function if high precision or reliability is needed.
	 */
	wall_clock_time_t wall_clock_time_now();
	
	/**
	 * Returns the time duration between @a now and @a past.
	 *
	 * @pre @a now must be later or at the same time as @a past, otherwise behavior is undefined.
	 *
	 * Convert the returned time into a time unit o choice by calling one of the @c convert_time_to
	 * functions.
	 */
	wall_clock_time_t duration( wall_clock_time_t const& now, 
								wall_clock_time_t const& past );
	
	
	bool operator==( wall_clock_time_t const& lhs, wall_clock_time_t const& rhs );	
	bool operator!=( wall_clock_time_t const& lhs, wall_clock_time_t const& rhs );
	bool operator<( wall_clock_time_t const& lhs, wall_clock_time_t const& rhs );
	bool operator<=( wall_clock_time_t const& lhs, wall_clock_time_t const& rhs );
	bool operator>( wall_clock_time_t const& lhs, wall_clock_time_t const& rhs );
	bool operator>=( wall_clock_time_t const& lhs, wall_clock_time_t const& rhs );
	
	
	/**
	 * Returns the time encoded in @a to_convert in milliseconds.
	 *
	 * @attention If @c T is to small an overflow and truncation can happen!
	 */
	template< typename T >
		T convert_time_to_ms( wall_clock_time_t const& to_convert )
		{
			T time_in_ms = static_cast< T >( to_convert.tv_sec * wall_clock_time_unit_factor_s_to_ms );
			time_in_ms += static_cast< T >( to_convert.tv_usec * wall_clock_time_unit_factor_us_to_ms );
			
			return time_in_ms;
		}
	
	
	/**
	 * Returns the time encoded in @a to_convert in seconds.
	 *
	 * @attention If @c T is to small an overflow and truncation can happen!
	 */
	template< typename T >
		T convert_time_to_s( wall_clock_time_t const& to_convert )
		{
			T time_in_s = static_cast< T >( to_convert.tv_sec );
			time_in_s += static_cast< T >( to_convert.tv_usec * wall_clock_time_unit_factor_us_to_s );
			
			return time_in_s;
		}	
	
	/**
	 * Returns the time encoded in @a to_convert in microseconds.
	 *
	 * @attention If @c T is to small an overflow and truncation can happen!
	 */
	template< typename T >
		T convert_time_to_us( wall_clock_time_t const& to_convert )
		{
			T time_in_us = static_cast< T >( to_convert.tv_sec * wall_clock_time_unit_factor_s_to_us );
			time_in_us += static_cast< T >( to_convert.tv_usec );
			
			return time_in_us;
		}	
	
} // namespace kapaga


#endif // KAPAGA_kapaga_wall_clock_time_H
