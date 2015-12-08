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
 * Thin wrapper around Posix rand_r.
 *
 * @attention Don't use these random number generators for serious business, like cryptography.
 */

#ifndef KAPAGA_kapaga_random_number_source_H
#define KAPAGA_kapaga_random_number_source_H

namespace kapaga {
	
	/**
	* A simple random number source.
	 *
	 * Mainly a thin wrapper around @c rand_r
	 *
	 * To reset call <code>set_source( random_number_source::default_seed() ) </code>.
	 */
	class random_number_source {
	public:
		typedef unsigned int source_type;
		typedef int value_type;
		
		explicit random_number_source( source_type seed = default_seed() );
		
		/**
			* Successive calls return a series of pseudo-random numbers.
		 *
		 * @return pseudo-random number based on @a source.
		 *
		 * @attention Not thread-safe/reentrant.
		 */
		value_type draw();
		
		source_type source() const;
		
		/**
			* @attention Not thread-safe/reentrant.
		 */
		void set_source( source_type source );
		
		static source_type default_seed();
		
		static value_type min();
		
		static value_type max();
		
		
		
	private:
			enum { default_seed_value = 1 };
		
		source_type source_;
	}; // class random_number_source	
	
	
} // namespace kapaga


#endif // KAPAGA_kapaga_random_number_source_H
