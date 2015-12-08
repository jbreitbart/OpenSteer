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
 * Detects the platform to include the right omp_stubs header.
 *
 * Header doesn't have header guards because the platform specific headers have them.
 */

// Include KAPAGA_POSIX
#include "kapaga/config.h"

#if defined( KAPAGA_POSIX )
	#include "kapaga/posix/omp_stubs.h"
#else
	#error Unknown platform.
#endif
