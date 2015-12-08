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
 * Macro to surpress warnings that function parameters aren't used.
 *
 * @todo Rename header to @c unused.h?
 */

#ifndef KAPAGA_unused_parameter_H
#define KAPAGA_unused_parameter_H

#if defined(KAPAGA_UNUSED_PARAMETER)
	#error unused_parameters redefines KAPAGA_UNUSED_PARAMETER 
#endif

#if defined(KAPAGA_UNUSED_VARIABLE)
	#error unused_parameters redefines KAPAGA_UNUSED_VARIABLE 
#endif

#if defined(KAPAGA_UNUSED_RETURN_VALUE)
	#error unused_parameters redefines KAPAGA_UNUSED_RETURN_VALUE 
#endif

/**
 * @todo Read C++ grammar and rename macro appropriate.
 */
/* 
#if defined(KAPAGA_UNUSED_EXPRESSION_VALUE)
	#error unused_parameters redefines KAPAGA_UNUSED_EXPRESSION_VALUE 
#endif
*/

/**
 * Macro to surpress warning that a function parameter isn't used.
 */
#define KAPAGA_UNUSED_PARAMETER(expr) (void)expr


/**
 * Macro to surpress warning that a variable isn't used.
 */
#define KAPAGA_UNUSED_VARIABLE(expr) (void)expr


/**
 * Macro to surpress warning that a return value of a function isn't used.
 */
#define KAPAGA_UNUSED_RETURN_VALUE(expr) (void)expr


/**
 * Macro to surpress warning that an expression value isn't used (though its side effects might be).
 *
 * @todo Read C++ grammar and rename macro appropriate.
 */
/* #define KAPAGA_UNUSED_EXPRESSION_VALUE(expr) (void)expr */


#endif /* KAPAGA_unused_parameter_H */
