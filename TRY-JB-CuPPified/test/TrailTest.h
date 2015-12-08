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
 * Unit test for @c OpenSteer::Trail.
 */
#ifndef OPENSTEER_TRAILTEST_H
#define OPENSTEER_TRAILTEST_H




#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestFixture.h>




namespace OpenSteer {
    
    
    class TrailTest : public CppUnit::TestFixture {
    public:
        TrailTest();
        virtual ~TrailTest();
        
        virtual void setUp();
        virtual void tearDown();
        
        CPPUNIT_TEST_SUITE(TrailTest);
        CPPUNIT_TEST(testConstruction);
        CPPUNIT_TEST(testRecordPosition);
        CPPUNIT_TEST(testTickDetection);
        CPPUNIT_TEST_SUITE_END();
        
    private:
        /**
         * Not implemented to make it non-copyable.
         */
        TrailTest( TrailTest const& );
        
        /**
         * Not implemented to make it non-copyable.
         */
        TrailTest& operator=( TrailTest const& );
        
    private:
        /**
         * Tests the different constructors.
         */
        void testConstruction();
        
        /**
         * Tests recording footsteps.
         */
        void testRecordPosition();
        
        /**
         * Test if footsteps that are farther than a tick duration away from
         * their predecessing footstep or that just took longer or as long as a
         * tick are detected accordingly.
         */
        void testTickDetection();
        
        // @todo Write insertion and tick detection tests for a Trail with just
        //       one footstep.
        
    }; // TrailTest
    
    
    
    
} // namespace OpenSteer


#endif // OPENSTEER_TRAILTEST_H
