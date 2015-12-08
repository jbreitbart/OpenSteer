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
 * Implements a compile time assertion facility as found in the Loki library
 * based on Modern C++ Design from Andrei Alexandrescu, section 2.1.
 *
 * @todo Rename file to compile_time_assert. It is not the class name that is important to the user
 *       but the macro name!
 */
#ifndef KAPAGA_kapaga_compile_time_assertion_H
#define KAPAGA_kapaga_compile_time_assertion_H


namespace kapaga {
    
    template< bool >  
    struct compile_time_assertion;
    
    
    template<>
        struct compile_time_assertion< true > {};
    
} // namespace kapaga


/**
 * Asserts at compile time that the compile time evaluable expression is @c true.
 * 
 * @a expr must be a compile time evalueable expression, like 
 * <code> sizeof( char ) <= sizeof( int )</code>.
 * @a msg must be a sentence describing the assertion which mustn't contain any
 * white spaces! Use underscores instead of whitespaces like in 
 * @c write_this_way.
 *
 * If @a expr is evaluated to @c true a specialized definition of the 
 * @c compile_time_assertion template class can be instantiated. A good compiler
 * will then remove the empty and non-used object.
 *
 * If @a expr is evaluated to @c false , if the assertion is wrong, no object
 * can be created because the definition is missing and the compiler will show
 * an error message containing the @a msg description.
 *
 * The typedef allows calling the macro even directly in a header or a source
 * file and surpresses warnings about unused variables in functions. Idea 
 * borrowed from the POSH library, see
 * http://www.bookofhook.com/poshlib/index.html
 */
#if defined(KAPAGA_COMPILE_TIME_ASSERT)
#error compile_time_assertion.h redefines KAPAGA_COMPILE_TIME_ASSERT 
#endif

#define KAPAGA_COMPILE_TIME_ASSERT(expr, msg) \
typedef kapaga::compile_time_assertion<(0!=(expr))> ERROR_##msg



#endif // KAPAGA_kapaga_compile_time_assertion_H
