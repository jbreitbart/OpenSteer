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
 * Assertion macro similar to assert from the C/C++ standard library.
 * Enable assertions by setting KAPAGA_DEBUG. If NDEBUG is set the assertions are
 * disabled even if KAPAGA_DEBUG is defined.
 *
 * @todo Should the logic be reversed and the macro should be names KAPAGA_NDEBUG? This way it would
 *       be enabled by default and the user immediatly sees (by crashes) if he violates assertions
 *       and has to disable them to get rid of them.
 */

#ifndef KAPAGA_assert_H
#define KAPAGA_assert_H

// Include assert
#include <cassert>


#if defined(KAPAGA_DEBUG)
	#define KAPAGA_ASSERT( x ) assert( x )
#else
	#define KAPAGA_ASSERT( x )
#endif


#endif // KAPAGA_assert_H
