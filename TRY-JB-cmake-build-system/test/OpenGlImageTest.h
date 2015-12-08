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
 * Unit test for @c OpenSteer::Graphics::OpenGlImage.
 */
#ifndef OPENSTEER_GRAPHICS_OPENGLIMAGETEST_H
#define OPENSTEER_GRAPHICS_OPENGLIMAGETEST_H




#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>


// Include OpenSteer::SharedArray
#include "OpenSteer/SharedArray.h"

// Include OpenSteer::Graphics::OpenGlImage
#include "OpenSteer/OpenGlImage.h"





namespace OpenSteer {
    
    namespace Graphics {
    
    
        class OpenGlImageTest : public CppUnit::TestFixture {
        public:
            OpenGlImageTest();
            virtual ~OpenGlImageTest();
            
            virtual void setUp();
            virtual void tearDown();
            
            CPPUNIT_TEST_SUITE(OpenGlImageTest);
            CPPUNIT_TEST(testConstruction);
            CPPUNIT_TEST(testComparison);
            CPPUNIT_TEST(testAssignment);
            CPPUNIT_TEST(testSwap);
            CPPUNIT_TEST(testSettingPixels);
            CPPUNIT_TEST(testClear);
            CPPUNIT_TEST(testCreatePixelColorElementFromAFloat);
            CPPUNIT_TEST_SUITE_END();
            
        private:
            /**
             * Not implemented to make it non-copyable.
             */
            OpenGlImageTest( OpenGlImageTest const& );
            
            /**
             * Not implemented to make it non-copyable.
             */
            OpenGlImageTest& operator=( OpenGlImageTest const& );
            
        private:
            /**
             *  Tests the different constructors.
             */
            void testConstruction();
            
            /**
             * Tests the comparisons operators.
             */
            void testComparison();
            
            /**
             * Test different assignment situations.
             */
            void testAssignment();
            
            /**
             * Explicit test of @c swap.
             */
            void testSwap();
            
            /**
             * Tests to set pixel values.
             */
            void testSettingPixels();
            
            /**
             * Tests calling clear for an @c OpenGlImage.
             */
            void testClear();
            
            
            /**
             * Test creating a pixel color element from a float.
             */
            void testCreatePixelColorElementFromAFloat();
            
        private:
            
            OpenGlImage testImage0;
            OpenGlImage testImage1;
            
        }; // class OpenGlImageTest
    
    
    } // namespace Graphics
    
} // namespace OpenSteer


#endif // OPENSTEER_GRAPHICS_OPENGLIMAGETEST_H
