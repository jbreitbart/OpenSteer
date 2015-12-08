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
 * Collection of random number generators that base their pseudo-random numbers on
 * @c random_number_source.
 *
 * @attention Don't use these random number generators for serious business, like cryptography.
 *
 * @todo Finish documentation.
 */

#ifndef KAPAGA_kapaga_randomizer_H
#define KAPAGA_kapaga_randomizer_H

// Include assert
#include <cassert>


// Include kapaga::random_number_source
#include "kapaga/random_number_source.h"


namespace kapaga {
	
	template< typename T >
	class randomizer;
	
	template<>
		class randomizer< int > {
		public:
			
			typedef int value_type;
			typedef random_number_source rand_source_type;
			
			/**
			 * @return a pseudo-random number between @c 0 and @c max() inclusive.
			 */
			static value_type draw( rand_source_type& source ) {
				return static_cast< value_type >( source.draw() );
			}
			
			/**
			 *
			 * @return a pseudo-random number between @c lower_bound and @c higher_bound inclusive.
			 *
			 * @pre <code> lower_bound <= higher_bound </code> otherwise behavior is undefined.
			 * @pre <code> higher_bound <= max() </code> otherwise behavior is undefined.			 
			 */
			static value_type draw( value_type lower_bound, 
									value_type higher_bound,
									rand_source_type& source ) {
				assert( lower_bound <= higher_bound &&
						"Error: lower_bound must be lesser or equal to higher_bound otherwise behavior is undefined" );
				assert( higher_bound <= max() &&
						"Error: higher_bound must be lesser or equal to max() otherwise behavior is undefined." );
				
				return lower_bound + ( source.draw() % ( higher_bound - lower_bound + value_type( 1 ) ) );
			}
			
			
			/**
			 * @return the minimum number @c draw without higher and lower boundary parameters 
			 *         generates.
			 */
			static value_type min() {
				return static_cast< value_type >( random_number_source::min() );
			}
			
			/**
			 * @return the maximum number @c draw without higher and lower boundary parameters 
			 *         generates.
			 */
			static value_type max() {
				return static_cast< value_type >( random_number_source::max() );
			}

		}; // class randomizer< int >
	
	
	template<>
		class randomizer< float > {
		public:
			typedef float value_type;
			typedef random_number_source source_type;
			
			
			/**
			 * @return a pseudo-random number between @c 0.0f and @c 1.0f.
			 */
			static value_type draw( source_type& source ) {
				return static_cast< value_type >( source.draw() ) 
				/  static_cast< value_type >( random_number_source::max() );
			}
			
			/**
			 * @return a pseudo-random number between @c lower_bound and @c higher_bound inclusive.
			 *
			 * @pre <code> lower_bound <= higher_bound </code> otherwise behavior is undefined.
			 */
			static value_type draw( value_type lower_bound, 
									value_type higher_bound,
									source_type& source ) {
				assert( lower_bound <= higher_bound && 
						"Error: lower_bound must be lesser or equal to higher_bound otherwise behavior is undefined." );
				
				return lower_bound + ( draw( source ) * ( higher_bound - lower_bound ) );
				
			}
			
			
			/**
			 * @return the minimum number @c draw without higher and lower boundary parameters 
			 *         generates.
			 */
			static value_type min()
			{
				return value_type( 0 );
			}
			
			/**
			 * @return the maximum number @c draw without higher and lower boundary parameters 
			 *         generates.
			 */
			static value_type max()
			{
				return value_type( 1 );
			}

			
		}; // class randomizer< float >
	
	
	
} // namespace kapaga



#endif // KAPAGA_kapaga_randomizer_H
