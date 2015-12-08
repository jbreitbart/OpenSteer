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
 * @file randomizer_utilities.h
 *
 * Collection of utilities and helper functions to use with randomizers.
 *
 * @todo Right name for this header?
 */

//Header guard macro format: KAPAGA_namespace_file_name_H
#ifndef KAPAGA_kapaga_randomizer_utilities_H
#define KAPAGA_kapaga_randomizer_utilities_H

// Include kapaga::random_number_source
#include "kapaga/random_number_source.h"

namespace kapaga {
	
	/**
	 * Uses @a source to draw a random number between @c -1.0f and @c 1.0f, most values should be
	 * around @c 0.0f and only a few values should be around @c -1.0f and @c 1.0f.
	 *
	 * Based on the random binomial from Ian Millington, Artificial Intelligence for Games, Morgan
	 * Kaufmann, 2006, p. 57.
	 */
	float binomial_randf( random_number_source& source );
	
	
} // namespace kapaga


#endif //  KAPAGA_kapaga_randomizer_utilities_H
