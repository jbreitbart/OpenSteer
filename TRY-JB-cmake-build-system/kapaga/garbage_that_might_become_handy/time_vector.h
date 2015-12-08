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
 * Representation of a time vector.
 *
 * @attention Implementation is not platform neutral. It uses the @c gettimeofday function mostly
 *            found on unix systems.
 *
 * @todo Write unit tests.
 * @todo Document possible overflow problems in conversions, duration calculations.
 * @todo Add @c elapsed_min and @c elapsed_max (see Boost thread library for inspiration).
 * @todo Add specialization on time type.
 */

#ifndef KAPAGA_kapaga_time_vector_H
#define KAPAGA_kapaga_time_vector_H

// Include gettimeofday, timeval
#include <sys/time.h>

// Include kapaga::time_unit
#include "kapaga/time_unit.h"


namespace kapaga {
		
	/**
	 * Type representation for the @c time_vector_zero token used to choose the @c time_vector 
	 * constructor that initilizes the time vector with zero.
	 */
	struct time_vector_zero_t {};
	
	/**
	 * Token used to choose the @c time_vector constructor initializing the time vector with zero.
	 */
	extern time_vector_zero_t const time_vector_zero;
	
	/**
	 * Type representation for the @c time_vector_now token used to choose the @c time_vector 
	 * constructor that initilizes the time vector with the current time.
	 */
	struct time_vector_now_t {};
	
	/**
	 * Token used to choose the @c time_vector constructor initializing the time vector with the 
	 * current time.
	 */
	extern time_vector_now_t const time_vector_now;
	
	
	/**
	 * A time vector represents a time point or a period of time.
	 *
	 * A time vector is composed of different time unit values, like a number of seconds or a
	 * number of microseconds. These values together build the duration of the time vector. The
	 * single values can be queried by the @c seconds, @c milliseconds, and @v microseconds
	 * member functions.
	 *
	 * It is possible that a time vector consists of enough microseconds that they accumulate to one
	 * milliseconds but only microseconds and no milliseconds are reported. This is based on the
	 * internal representation.
	 * Use one of the @c duration functions to get the duration in a special time unit represented
	 * by a numerical type (most of the time an integer time, for example @c long).
	 *
	 * @c T should be a primitive type, like @c long, or @c double. Choose the type based on the
	 * precision needed and the periods of time the time vector should represent (the longer the
	 * time period the greater the size of the type).
	 *
	 *
	 @attention Most of the class member functions are safe to use from a single thread, however
	 *            some functions are annotated to be eventually not thread safe if they are called
	 *            for different instances but from different threads!
	 *
	 *
	 *
	 *
	 * Time in computers is a strange concept. Typically a function is queried to return the current 
	 * time. However the returned time value does only make sense relative to a time point in the 
	 * past.
	 * You can measeure the period between these two times.
	 *
	 * If you take the time at a later moment you can calculate the relative difference between
	 * the two time points to get the duration between them.
	 * 
	 * To keep the maximum resolution this calculation should be done using the time value time type 
	 * which is then finaly converted to an integer type to represent day, hours, minutes, seconds, 
	 * milliseconds, microsecons, nanoseconds, or whatever time unit is needed by the user.
	 *
	 * Therefore the time type is used to represent a time type and a (high resolution) time period.
	 * This reminds of the difference of points and vectors in 3D graphics. Typical implementations
	 * for this field just provide a 3-tuple or 4-tuple called vector, which can represent 3D points
	 * and 3D vectors. The idea is not to think in 3D or geometry terms but in terms of an algebra
	 * whose vectors can be used as points or vectors in the field of 3D or geometry.
	 *
	 * Based on this example Kapaga provides a time vector to represent time points and time 
	 * durations. However the parallel isn't perfect because there are no operations other than
	 * @c duration_in_seconds, etc. on time vectors.
	 *
	 * @todo Rename to @c time?
	 */
	template< typename Time = standard_clock_time >
	class time_vector : private Time {
	public:
		/**
		 * Type used to represent the components of the time vector.
		 */
		typedef typename Time time_type;
		
		using time_type::duration_type;
		using time_type::clock_tick_type;
		using time_type::clock_ticks_per_second_type;
		
		/**
		 * Constructs a time vector without initializing it.
		 */
		inline time_vector();
	
		/**
		 * Construct a time vector for the current time (now).
		 *
		 * @example <code>time_vector tv( time_now );</code>
		 *
		 * @attention This member function is eventually not thread safe because it has to call a 
		 *            system function to get the current time which might not be thread safe.
		 */
		inline explicit time_vector( time_vector_now_t const& );
	
		
		/**
		 * Constructs a zero time vector.
		 *
		 * @example <code>time_vector tv( time_zero );</code>
		 */ 
		inline explicit time_vector( time_vector_zero_t const& );
		
		
		/**
		 * @return The number of seconds (aside the number of milliseconds, etc.) composing the 
		 *         time period.
		 */
		using time_type::seconds();
		
		/**
		 * @return The number of milliseconds (aside the number seconds, etc.) composing the 
		 *         time period.
		 */
		using time_type::milliseconds;
		
		/**
		 * @return The number of microseconds (aside the number of seconds, etc.) composing the 
		 *         time period.
		 */
		using time_type::microseconds;
		
		/**
		 * @return The number of nanoseconds (aside the number of seconds, etc.) composing the 
		 *         time period.
		 */
		using time_type::nanoseconds;

		
		using time_type::clock_ticks;
		
		using time_type::clock_ticks_per_second; 
		
		/**
		 * It is possible to querey for the nanoseconds/microseconds/milliseconds part of the time 
		 * vector but based on the system function used internally they might not be supported.
		 *
		 * @return the highest time resolution supported.
		 */
		using time_type::highest_resolution;
		
		using time_type::measures_clock_ticks;
		
		
		using time_type::operator+=;
		
		using time_type::operator-;
		
		using time_type::operator-=;
		
		using time_type::normalize;
		
		 
		/**
		 * Sets the time vector to the current time. 
		 *
		 * @attention This member function is eventually not thread safe because it has to call a 
		 *            system function to get the current time which might not be thread safe.
		 */
		inline void assign( time_vector_now_t );
		
		/**
		 * Sets the time vector to zero.
		 */
		inline void assign( time_vector_zero_t );
		
		
	}; // class time_vector
	
	
	// @todo Is a template parameter really needed? Why don't just the conversion/duration routines
	//       have a template parameter for their destination type?
	template< typename T = double >
	struct omp_time {
		typedef double time_type;
		typedef T duration_type;
		typedef T clock_tick_type;
		typedef T clock_ticks_per_second_type;
		
		
		omp_time() : value_( 0 ) {}
		explicit omp_time( time_type t ) : value_( t ) {}
		
		static omp_time now() {
			return omp_time( omp_get_wtime() );
		}
		
		
		duration_type seconds() const {
			return static_cast< duration_type >( value_ );
		}
		
		duration_type milliseconds() const {
			return static_cast< duration_type >( 0 );
		}
		
		duration_type convert_to_microseconds() const {
			return static_cast< duration_type >( 0 );
		}
		
		duration_type convert_to_nanoseconds() const {
			return static_cast< duration_type >( 0 );
		}
		
		clock_tick_type clock_ticks() const {
			return static_cast< clock_tick_type >( 0 );
		}
		
		static clock_ticks_per_second_type clock_ticks_per_second() {
			return static_cast< clock_ticks_per_second_type >( 1 );
		}
		
		static time_unit highest_resolution() const {
			return seconds;
		}
		
		/**
		 * @attention If a time is measured in clock ticks the other components won't carry any
		 *            value. If the time is not measured in clock ticks @c clock_ticks returns
		 *            @c 0 and @c clock_ticks_per_seconds returns @c 1.
		 */
		static bool measures_clock_ticks() {
			return true;
		}
		
		
		standard_clock_time& operator-=( standard_clock_time const& other ) {
			value_ -= other.value_;
			return *this;
		}
		
		void normalize() {}
		
		
		time_type value_;
	};
	
	// @todo Add a capability symbol class that allows to detect when a time class needs 
	//       normalization to be able to write @c operator< more efficient.
	// @todo Is a template parameter really needed? Why don't just the conversion/duration routines
	//       have a template parameter for their destination type?
	template< typename T = long >
	struct posix_time {
		typedef timeval time_type;
		typedef T duration_type;
		typedef T clock_tick_type;
		typedef T clock_ticks_per_second_type;
		typedef double normalization_type; // To normalize the nanoseconds after @c op-=
		
		posix_time() { value_ = { 0, 0 }; }
		
		
		static posix_time now() {
			posix_time t;
			// @todo Check if @c gettimeofday returns a nanosecond value that isn't equal to one or
			//       more seconds.
			// @todo Do I need error handling here?
			int const ignore = gettimeofday( &t.value_, 0 );
			return t;
		}
		
		
		duration_type seconds() const {
			return static_cast< duration_type >( value_.tv_sec );
		}
		
		duration_type milliseconds() const {
			return static_cast< duration_type >( 0 );
		}
		
		duration_type convert_to_microseconds() const {
			return static_cast< duration_type >( value_.tv_usec );
		}
		
		duration_type convert_to_nanoseconds() const {
			return static_cast< duration_type >( 0 );
		}
		
		clock_tick_type clock_ticks() const {
			return static_cast< clock_tick_type >( 0 );
		}
		
		static clock_ticks_per_second_type clock_ticks_per_second() {
			return static_cast< clock_ticks_per_second_type >( 1 );
		}
		
		static time_unit highest_resolution() const {
			return microseconds;
		}
		
		// @todo Rename to @c measures_cpu_ticks.
		static bool measures_clock_ticks() {
			return false;
		}
		
		/**
		 *
		 * @attention Call normalize after using @c operator-= or using @c operator-= in a sequence.
		 */
		standard_clock_time& operator-=( standard_clock_time const& other ) {
			value_.tv_sec -= other.value_.tv_sec;
			// @todo Add test and correction if the microseconds are equivalent to more or equal to
			//       a second.
			value_.tv_usec -= other.value_.tv_usec;
			
			return *this;
		}
		
		void normalize() {
			// @todo overflow / underflow is problematic!
			normalization_type in_sec = static_cast< normalization_type >( value_.tv_usec ) / static_cast< normalization_type >( 1000 * 1000 );
			
			// @todo Use the right @c floor function.
			normalization_type sec = floor( in_sec );
			normalization_type usec = in_sec - sec;
			value_.tv_sec += static_cast< duration_type >( sec );
			value_.tv_usec = static_cast< duration_type >( usec );	
		}
		
		
		
		
		time_type value_;
	};
	
	
	// @todo Is a template parameter really needed? Why don't just the conversion/duration routines
	//       have a template parameter for their destination type?
	// @todo What about an error symbolizing time (read standard)?
	template< typename T = double >
	struct standard_clock_time {
		typedef std::clock_t time_type;
		typedef T duration_type;
		typedef std::clock_t clock_tick_type;
		typedef int clock_ticks_per_second_type;
		
		standard_clock_time() : value_( 0 ) {}
		explicit standard_clock_time( time_type t ) : value_( t ) {}
		
		static standard_clock_time now() {
			return standard_clock_time( std::clock() );
		}
		
		duration_type seconds() const {
			return static_cast< duration_type >( value_.tv_sec );
		}
		
		duration_type milliseconds() const {
			return static_cast< duration_type >( 0 );
		}
		
		duration_type microseconds() const {
			return static_cast< duration_type >( 0 );
		}
		
		duration_type nanoseconds() const {
			return static_cast< duration_type >( 0 );
		}			
		
		
		clock_tick_type clock_ticks() const {
			return value_;
		}
		
		static clock_ticks_per_second_type clock_ticks_per_second() {
			return CLOCKS_PER_SEC;
		}
		
		
		static time_unit highest_resolution() const {
			return seconds;
		}
		
		static bool measures_clock_ticks() {
			return false;
		}
		
		standard_clock_time& operator-=( standard_clock_time const& other ) {
			value_ -= other.value_;
			return *this;
		}
		
		void normalize() {}
		
		time_type value_;
	};
	
	
	// @todo Just use a template parameter for the value to convert to.
	template< typename T >
		inline time_vector< T >::duration_type convert_to_seconds( time_vector< T > const& tv ) {
			typedef time_vector< T >::duration_type duration_type;
			return tv.seconds() + 
				tv.milliseconds() / static_cast< duration_type >( 1000 ) +
				tv.microseconds() / static_cast< duration_type >( 1000 * 1000 ) +
				tv.nanoseconds() / static_cast< duration_type >( 1000 * 1000 * 1000 );
		}
	
	// @todo Just use a template parameter for the value to convert to.
	template< typename T >
		inline time_vector< T >::duration_type convert_to_milliseconds( time_vector< T > const& tv ) {
			typedef time_vector< T >::duration_type duration_type;
			return tv.seconds() * static_cast< duration_type >( 1000 ) + 
				tv.milliseconds()  +
				tv.microseconds() / static_cast< duration_type >( 1000 ) +
				tv.nanoseconds() / static_cast< duration_type >( 1000 * 1000 ); 
		}
	
	// @todo Just use a template parameter for the value to convert to.
	template< typename T >
		inline time_vector< T >::duration_type convert_to_microseconds( time_vector< T > const& tv ) {
			typedef time_vector< T >::duration_type duration_type;
			return tv.seconds()  * static_cast< duration_type >( 1000 * 1000 ) +
				tv.milliseconds() * static_cast< duration_type >( 1000 ) +
				tv.microseconds()  +
				tv.nanoseconds() / static_cast< duration_type >( 1000 );
		}
	
	// @todo Just use a template parameter for the value to convert to.
	template< typename T >
		inline time_vector< T >::duration_type convert_to_nanoseconds( time_vector< T > const& tv ) {
			typedef time_vector< T >::duration_type duration_type;
			return tv.seconds() * static_cast< duration_type >( 1000 * 1000 * 1000 ) + 
				tv.milliseconds() * static_cast< duration_type >( 1000 * 1000 ) +
				tv.microseconds() * static_cast< duration_type >( 1000 ) +
				tv.nanoseconds();
		}
	
	
	
	template< typename T >
		bool operator==( time_vector< T > const& lhs, time_vector< T > const& rhs ) {
			return lhs.clock_ticks() == rhs.clock_ticks &&
				lhs.clock_ticks_per_second() == rhs.clock_ticks_per_seconds() &&
				lhs.seconds() == rhs.seconds() &&
				lhs.milliseconds() == rhs.milliseconds() &&
				lhs.microseconds() == rhs.microseconds() &&
				lhs.nanoseconds() == rhs.nanoseconds():
		}
	
	template< typename T >
		bool operator!=( time_vector< T > const& lhs, time_vector< T > const& rhs ) {
			return ! operator==( lhs, rhs );
		}
	
	template< typename T >
		bool operator<( time_vector< T >  lhs, time_vector< T >  rhs ) {
			typedef time_vector< T >::duration_type duration_type;
			
			lhs.normalize();
			rhs.normalize();
			
			if ( time_vector< T >::measures_clock_time() ) {
				if ( static_cast< duration_type >( lhs.clock_ticks() ) / static_cast< duration_type>( lhs.clock_ticks_per_second() ) < 
					 static_cast< duration_type >( rhs.clock_ticks() ) / static_cast< duration_type>( rhs.clock_ticks_per_second() ) ) {
					return true;
				} else {
					return false;
				}
				
			} else {
			
			
				if ( lhs.seconds() < rhs.seconds() ) {
					return true;	
				} else if ( lhs.seconds() > rhs.seconds() ) {
					return false;
				}
				
				if ( lhs.milliseconds() < rhs.milliseconds() ) {
					return true;
				} else if ( lhs.milliseconds() > rhs.milliseconds() ) {
					return false;
				}
				
				if ( lhs.microseconds() < rhs.microseconds() ) {
					return true;
				} else if ( lhs.microseconds() > rhs.microseconds() ) {
					return false;
				}
				
				if ( lhs.nanosecons() < rhs.nanoseconds() ) {
					return true
				}
			}
			
			return false;
		}
	
	
	template< typename T >
		bool operator>( time_vector< T > const& lhs, time_vector< T > const& rhs ) {
			return rhs < lhs;
		}
	
	
	template< typename T >
		bool operator<=( time_vector< T > const& lhs, time_vector< T > const& rhs ) {
			return ! lhs > rhs;
		}
	
	
	template< typename T >
		bool operator>=( time_vector< T > const& lhs, time_vector< T > const& rhs ) {
			return rhs <= lhs;
		}
	
	
	// @todo Just use a template parameter for the value to convert to.
	template< typename T >
		inline time_vector< T >::duration_type duration_in_seconds( time_vector< T > const& lhs, time_vector< T > const& rhs ) {
			typedef time_vector< T >::duration_type duration_type;
			
			return ( ( lhs.seconds() - rhs.seconds() ) ) +
				( ( lhs.milliseconds() - rhs.milliseconds() ) / static_cast< duration_type >( 1000 ) ) +
				( ( lhs.microseconds() - rhs.microseconds() ) / static_cast< duration_type >( 1000 * 1000 ) ) +
				( ( lhs.nanoseconds() - rhs.nanoseconds() ) / static_cast< duration_type >( 1000 * 1000 * 1000 ) );
			
		}
	
	// @todo Just use a template parameter for the value to convert to.
	template< typename T >
		inline time_vector< T >::duration_type duration_in_milliseconds( time_vector< T > const& lhs, time_vector< T > const& rhs ) {
			typedef time_vector< T >::duration_type duration_type;
			
			return ( ( lhs.seconds() - rhs.seconds() ) * static_cast< duration_type >( 1000 ) )+
				( ( lhs.milliseconds() - rhs.milliseconds() ) ) +
				( ( lhs.microseconds() - rhs.microseconds() ) / static_cast< duration_type >( 1000  ) ) +
				( ( lhs.nanoseconds() - rhs.nanoseconds() ) / static_cast< duration_type >( 1000 * 1000 ) );
			
		}
	
	// @todo Just use a template parameter for the value to convert to.
	template< typename T >
		inline time_vector< T >::duration_type duration_in_microseconds( time_vector< T > const& lhs, time_vector< T > const& rhs ) {
			typedef time_vector< T >::duration_type duration_type;
			
			return ( ( lhs.seconds() - rhs.seconds() ) * static_cast< duration_type >( 1000 * 1000 ) )+
				( ( lhs.milliseconds() - rhs.milliseconds() ) * static_cast< duration_type >( 1000 ) ) +
				( ( lhs.microseconds() - rhs.microseconds() ) ) +
				( ( lhs.nanoseconds() - rhs.nanoseconds() ) / static_cast< duration_type >( 1000 ) );
			
		}
	
	// @todo Just use a template parameter for the value to convert to.
	template< typename T >
		inline time_vector< T >::duration_type duration_in_nanoseconds( time_vector< T > const& lhs, time_vector< T > const& rhs ) {
			typedef time_vector< T >::duration_type duration_type;
			
			return ( ( lhs.seconds() - rhs.seconds() ) * static_cast< duration_type >( 1000 * 1000 * 1000 ) )+
				( ( lhs.milliseconds() - rhs.milliseconds() ) * static_cast< duration_type >( 1000 * 1000 ) ) +
				( ( lhs.microseconds() - rhs.microseconds() ) * static_cast< duration_type >( 1000 ) ) +
				( ( lhs.nanoseconds() - rhs.nanoseconds() ) );
			
		}
	
	
	
	

	
	
} // namespace kapaga
	
#endif // KAPAGA_kapaga_time_vector_H
