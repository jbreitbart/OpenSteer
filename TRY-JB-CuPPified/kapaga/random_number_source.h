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
 * A random number source. 
 *
 * Implementation is platform dependent and done in platform specific sub directories.
 *
 * No header guards are used because the headers included from here have the same name as this
 * header and use header guards themselves.
 */

// Include KAPAGA_POSIX
#include "kapaga/config.h"


#if defined(KAPAGA_POSIX)
	#include "kapaga/posix/random_number_source.h"
#else
	#error Unknown platform.
#endif

