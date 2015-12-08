/**
 * OpenSteer -- Steering Behaviors for Autonomous Characters
 *
 * Copyright (c) 2002-2005, Sony Computer Entertainment America
 * Original author: Craig Reynolds <craig_reynolds@playstation.sony.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *
 * @file
 *
 * @author Bjoern Knafla <bknafla@uni-kassel.de>
 *
 * Implements a compile time assertion facility as found in the Loki library
 * based on Modern C++ Design from Andrei Alexandrescu, section 2.1.
 *
 * @todo Rewrite the macro code to be cleaner and more useable.
 */
#ifndef OPENSTEER_COMPILETIMEASSERTION_H
#define OPENSTEER_COMPILETIMEASSERTION_H


namespace OpenSteer {
    
    template< bool >  
    struct CompileTimeAssertion;
    
    
    template<>
        struct CompileTimeAssertion< true > {};
    
} // namespace OpenSteer


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
 * @c CompileTimeAssertion template class can be instantiated. A good compiler
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
#define OPENSTEER_COMPILE_TIME_ASSERT(expr, msg) \
typedef OpenSteer::CompileTimeAssertion<(0!=(expr))> ERROR_##msg



#endif // OPENSTEER_COMPILETIMEASSERTION_H
