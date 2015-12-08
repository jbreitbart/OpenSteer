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
 * Unit test for @c OpenSteer::SharedPointer.
 */
#include "TrailTest.h"

// Include OpenSteer::Trail
#include "OpenSteer/Trail.h"

// Include OpenSteer::size_t
#include "OpenSteer/StandardTypes.h"

// Include OpenSteer::Vec3
#include "OpenSteer/Vec3.h"


CPPUNIT_TEST_SUITE_REGISTRATION( OpenSteer::TrailTest );



OpenSteer::TrailTest::TrailTest()
{
    // Nothing to do.
}



OpenSteer::TrailTest::~TrailTest()
{
    // Nothing to do.
}




void 
OpenSteer::TrailTest::setUp()
{
    TestFixture::setUp();
}



void 
OpenSteer::TrailTest::tearDown()
{
    TestFixture::tearDown();
}




void
OpenSteer::TrailTest::testConstruction()
{
    // Construct with standard constructor
    Trail< 4 > trail0;
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 0 ), trail0.footstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 0 ), trail0.positionCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 ), trail0.maxFootstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 * 2 ), trail0.maxPositionCount() );
    CPPUNIT_ASSERT_EQUAL( 5.0f, trail0.duration() );
    CPPUNIT_ASSERT( trail0.empty() );
    
    // Construct with constructor taking a duration argument.
    Trail< 4 > trail1( 10.0f );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 0 ), trail1.footstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 0 ), trail1.positionCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 ), trail1.maxFootstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 * 2 ), trail1.maxPositionCount() );
    CPPUNIT_ASSERT_EQUAL( 10.0f, trail1.duration() );
    CPPUNIT_ASSERT( trail1.empty() );
    
    // Test copy-constructor.
    Trail< 4 > trail2(trail0 );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 0 ), trail2.footstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 0 ), trail2.positionCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 ), trail2.maxFootstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 * 2 ), trail2.maxPositionCount() );
    CPPUNIT_ASSERT_EQUAL( trail0.duration(), trail2.duration() );
    CPPUNIT_ASSERT( trail2.empty() );
    
    Trail< 4 > trail3(trail1 );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 0 ), trail3.footstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 0 ), trail3.positionCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 ), trail3.maxFootstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 * 2 ), trail3.maxPositionCount() );
    CPPUNIT_ASSERT_EQUAL( 10.0f, trail3.duration() );
    CPPUNIT_ASSERT( trail3.empty() );
    
    
    // Test assignment-operator.
    trail3 = trail0;
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 0 ), trail3.footstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 0 ), trail3.positionCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 ), trail3.maxFootstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 * 2 ), trail3.maxPositionCount() );
    CPPUNIT_ASSERT_EQUAL( 5.0f, trail3.duration() );
    CPPUNIT_ASSERT( trail3.empty() );
    
    trail3 = trail1;
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 0 ), trail3.footstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 0 ), trail3.positionCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 ), trail3.maxFootstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 * 2 ), trail3.maxPositionCount() );
    CPPUNIT_ASSERT_EQUAL( 10.0f, trail3.duration() );
    CPPUNIT_ASSERT( trail3.empty() );
    
}



void
OpenSteer::TrailTest::testRecordPosition() 
{
    Vec3 const position0( 1.0f, 0.0f, 0.0f );
    Vec3 const position1( 2.0f, 0.0f, 0.0f );
    Vec3 const position2( 3.0f, 0.0f, 0.0f );
    Vec3 const position3( 4.0f, 0.0f, 0.0f );
    Vec3 const position4( 5.0f, 0.0f, 0.0f );
    Vec3 const position5( 6.0f, 0.0f, 0.0f );
    
    float const time0( 1.0f );
    float const time1( 1.5f );
    float const time2( 2.0f );
    float const time3( 2.1f );
    float const time4( 2.5f );
    float const time5( 2.7f );
    
    // Add positions to a trail with a long trail duration so positions won't
    // get pruned automatically.
    Trail< 2 > trail0( 100.0f );
    
    trail0.recordPosition( position0, time0 );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 0 ), trail0.footstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 0 ), trail0.positionCount() );
    
    trail0.recordPosition( position1, time1 );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 1 ), trail0.footstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 2 ), trail0.positionCount() );
    CPPUNIT_ASSERT_EQUAL( position0, trail0.footstepPosition( 0 ) );
    CPPUNIT_ASSERT_EQUAL( position0, trail0.footstepPositionAtTime( 0 ) );
    CPPUNIT_ASSERT_EQUAL( position1, trail0.footstepPosition( 1 ) );
    CPPUNIT_ASSERT_EQUAL( position1, trail0.footstepPositionAtTime( 1 ) );
    CPPUNIT_ASSERT_EQUAL( time0, trail0.footstepPositionTime( 0 ) );
    CPPUNIT_ASSERT_EQUAL( time0, trail0.footstepPositionTimeAtTime( 0 ) );
    CPPUNIT_ASSERT_EQUAL( time1, trail0.footstepPositionTime( 1 ) );
    CPPUNIT_ASSERT_EQUAL( time1, trail0.footstepPositionTimeAtTime( 1 ) );
    
    trail0.recordPosition( position2, time2 );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 1 ), trail0.footstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 2 ), trail0.positionCount() );
    CPPUNIT_ASSERT_EQUAL( position0, trail0.footstepPosition( 0 ) );
    CPPUNIT_ASSERT_EQUAL( position0, trail0.footstepPositionAtTime( 0 ) );
    CPPUNIT_ASSERT_EQUAL( position1, trail0.footstepPosition( 1 ) );
    CPPUNIT_ASSERT_EQUAL( position1, trail0.footstepPositionAtTime( 1 ) );
    CPPUNIT_ASSERT_EQUAL( time0, trail0.footstepPositionTime( 0 ) );
    CPPUNIT_ASSERT_EQUAL( time0, trail0.footstepPositionTimeAtTime( 0 ) );
    CPPUNIT_ASSERT_EQUAL( time1, trail0.footstepPositionTime( 1 ) );
    CPPUNIT_ASSERT_EQUAL( time1, trail0.footstepPositionTimeAtTime( 1 ) );
    
    trail0.recordPosition( position3, time3 );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 2 ), trail0.footstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 ), trail0.positionCount() );
    CPPUNIT_ASSERT_EQUAL( position0, trail0.footstepPosition( 0 ) );
    CPPUNIT_ASSERT_EQUAL( position0, trail0.footstepPositionAtTime( 0 ) );
    CPPUNIT_ASSERT_EQUAL( position1, trail0.footstepPosition( 1 ) );
    CPPUNIT_ASSERT_EQUAL( position1, trail0.footstepPositionAtTime( 1 ) );
    CPPUNIT_ASSERT_EQUAL( position2, trail0.footstepPosition( 2 ) );
    CPPUNIT_ASSERT_EQUAL( position2, trail0.footstepPositionAtTime( 2 ) );
    CPPUNIT_ASSERT_EQUAL( position3, trail0.footstepPosition( 3 ) );
    CPPUNIT_ASSERT_EQUAL( position3, trail0.footstepPositionAtTime( 3 ) );    
    CPPUNIT_ASSERT_EQUAL( time0, trail0.footstepPositionTime( 0 ) );
    CPPUNIT_ASSERT_EQUAL( time0, trail0.footstepPositionTimeAtTime( 0 ) );
    CPPUNIT_ASSERT_EQUAL( time1, trail0.footstepPositionTime( 1 ) );
    CPPUNIT_ASSERT_EQUAL( time1, trail0.footstepPositionTimeAtTime( 1 ) );
    CPPUNIT_ASSERT_EQUAL( time2, trail0.footstepPositionTime( 2 ) );
    CPPUNIT_ASSERT_EQUAL( time2, trail0.footstepPositionTimeAtTime( 2 ) );
    CPPUNIT_ASSERT_EQUAL( time3, trail0.footstepPositionTime( 3 ) );
    CPPUNIT_ASSERT_EQUAL( time3, trail0.footstepPositionTimeAtTime( 3 ) );
    
    
    trail0.recordPosition( position4, time4 );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 2 ), trail0.footstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 ), trail0.positionCount() );
    CPPUNIT_ASSERT_EQUAL( position0, trail0.footstepPosition( 0 ) );
    CPPUNIT_ASSERT_EQUAL( position0, trail0.footstepPositionAtTime( 0 ) );
    CPPUNIT_ASSERT_EQUAL( position1, trail0.footstepPosition( 1 ) );
    CPPUNIT_ASSERT_EQUAL( position1, trail0.footstepPositionAtTime( 1 ) );
    CPPUNIT_ASSERT_EQUAL( position2, trail0.footstepPosition( 2 ) );
    CPPUNIT_ASSERT_EQUAL( position2, trail0.footstepPositionAtTime( 2 ) );
    CPPUNIT_ASSERT_EQUAL( position3, trail0.footstepPosition( 3 ) );
    CPPUNIT_ASSERT_EQUAL( position3, trail0.footstepPositionAtTime( 3 ) );    
    CPPUNIT_ASSERT_EQUAL( time0, trail0.footstepPositionTime( 0 ) );
    CPPUNIT_ASSERT_EQUAL( time0, trail0.footstepPositionTimeAtTime( 0 ) );
    CPPUNIT_ASSERT_EQUAL( time1, trail0.footstepPositionTime( 1 ) );
    CPPUNIT_ASSERT_EQUAL( time1, trail0.footstepPositionTimeAtTime( 1 ) );
    CPPUNIT_ASSERT_EQUAL( time2, trail0.footstepPositionTime( 2 ) );
    CPPUNIT_ASSERT_EQUAL( time2, trail0.footstepPositionTimeAtTime( 2 ) );
    CPPUNIT_ASSERT_EQUAL( time3, trail0.footstepPositionTime( 3 ) );
    CPPUNIT_ASSERT_EQUAL( time3, trail0.footstepPositionTimeAtTime( 3 ) );
    
    
    trail0.recordPosition( position5, time5 );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 2 ), trail0.footstepCount() );
    CPPUNIT_ASSERT_EQUAL( static_cast< size_t >( 4 ), trail0.positionCount() );
    CPPUNIT_ASSERT_EQUAL( position4, trail0.footstepPosition( 0 ) );
    CPPUNIT_ASSERT_EQUAL( position2, trail0.footstepPositionAtTime( 0 ) );
    
    CPPUNIT_ASSERT_EQUAL( position5, trail0.footstepPosition( 1 ) );
    CPPUNIT_ASSERT_EQUAL( position3, trail0.footstepPositionAtTime( 1 ) );
    
    CPPUNIT_ASSERT_EQUAL( position2, trail0.footstepPosition( 2 ) );
    CPPUNIT_ASSERT_EQUAL( position4, trail0.footstepPositionAtTime( 2 ) );
    
    CPPUNIT_ASSERT_EQUAL( position3, trail0.footstepPosition( 3 ) );
    CPPUNIT_ASSERT_EQUAL( position5, trail0.footstepPositionAtTime( 3 ) ); 
    
    
    CPPUNIT_ASSERT_EQUAL( time4, trail0.footstepPositionTime( 0 ) );
    CPPUNIT_ASSERT_EQUAL( time2, trail0.footstepPositionTimeAtTime( 0 ) );
    
    CPPUNIT_ASSERT_EQUAL( time5, trail0.footstepPositionTime( 1 ) );
    CPPUNIT_ASSERT_EQUAL( time3, trail0.footstepPositionTimeAtTime( 1 ) );
    
    CPPUNIT_ASSERT_EQUAL( time2, trail0.footstepPositionTime( 2 ) );
    CPPUNIT_ASSERT_EQUAL( time4, trail0.footstepPositionTimeAtTime( 2 ) );
    
    CPPUNIT_ASSERT_EQUAL( time3, trail0.footstepPositionTime( 3 ) );
    CPPUNIT_ASSERT_EQUAL( time5, trail0.footstepPositionTimeAtTime( 3 ) );    
 
}



void
OpenSteer::TrailTest::testTickDetection()
{
    float const tick( 1.0f );
    
    Vec3 const position0( 1.0f, 0.0f, 0.0f );
    Vec3 const position1( 2.0f, 0.0f, 0.0f );
    
    Vec3 const position2( 3.0f, 0.0f, 0.0f );
    Vec3 const position3( 4.0f, 0.0f, 0.0f );
    
    Vec3 const position4( 5.0f, 0.0f, 0.0f );
    Vec3 const position5( 6.0f, 0.0f, 0.0f );
    
    Vec3 const position6( 7.0f, 0.0f, 0.0f );
    Vec3 const position7( 8.0f, 0.0f, 0.0f );
    
    Vec3 const position8( 9.0f, 0.0f, 0.0f );
    Vec3 const position9( 10.0f, 0.0f, 0.0f );
    
    float const time0( 1.0f );
    float const time1( 1.5f );
    
    float const time2( 2.0f );
    float const time3( 2.1f );
    
    float const time4( 2.5f );
    float const time5( 3.5f );
    
    float const time6( 3.7f );
    float const time7( 4.0f );
    
    float const time8( 5.0f );
    float const time9( 5.7f );
    
    Trail< 5 > trail0;
    trail0.recordPosition( position0, time0 );
    trail0.recordPosition( position1, time1 );
    trail0.recordPosition( position2, time2 );
    trail0.recordPosition( position3, time3 );
    trail0.recordPosition( position4, time4 );
    trail0.recordPosition( position5, time5 );
    trail0.recordPosition( position6, time6 );
    trail0.recordPosition( position7, time7 );
    trail0.recordPosition( position8, time8 );
    trail0.recordPosition( position9, time9 );
    
    CPPUNIT_ASSERT( ! trail0.footstepAtTick( 0, tick ) );
    CPPUNIT_ASSERT( ! trail0.footstepAtTick( 1, tick ) );
    CPPUNIT_ASSERT(   trail0.footstepAtTick( 2, tick ) );
    CPPUNIT_ASSERT( ! trail0.footstepAtTick( 3, tick ) );
    CPPUNIT_ASSERT(   trail0.footstepAtTick( 4, tick ) );
    
    Trail< 5 > trail1;
    trail1.recordPosition( Vec3( 0.0f, 0.0f, 0.0f ), 0.0f );
    trail1.recordPosition( Vec3( 0.0f, 0.0f, 0.0f ), 0.0f );
    
    trail1.recordPosition( position0, time0 );
    trail1.recordPosition( position1, time1 );
    trail1.recordPosition( position2, time2 );
    trail1.recordPosition( position3, time3 );
    trail1.recordPosition( position4, time4 );
    trail1.recordPosition( position5, time5 );
    trail1.recordPosition( position6, time6 );
    trail1.recordPosition( position7, time7 );
    trail1.recordPosition( position8, time8 );
    trail1.recordPosition( position9, time9 );
    
    CPPUNIT_ASSERT( ! trail1.footstepAtTickAtTime( 0, tick ) );
    CPPUNIT_ASSERT( ! trail1.footstepAtTickAtTime( 1, tick ) );
    CPPUNIT_ASSERT(   trail1.footstepAtTickAtTime( 2, tick ) );
    CPPUNIT_ASSERT( ! trail1.footstepAtTickAtTime( 3, tick ) );
    CPPUNIT_ASSERT(   trail1.footstepAtTickAtTime( 4, tick ) );
}



