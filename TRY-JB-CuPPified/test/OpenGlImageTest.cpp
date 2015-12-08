#include "OpenGlImageTest.h"


// Include std::numerical_limits
#include <limits>

// Include std::cout, std::endl
#include <iostream>

// Include std::copy
#include <algorithm>





// Include OpenSteer::size_t
#include "OpenSteer/StandardTypes.h"





CPPUNIT_TEST_SUITE_REGISTRATION( OpenSteer::Graphics::OpenGlImageTest );




namespace {
    
    using namespace OpenSteer;
    using namespace OpenSteer::Graphics;
    
    OpenSteer::size_t const testImage0Width = 2;
    OpenSteer::size_t const testImage0Height = 3;
    OpenGlImage::pixel_color_element_type imageData0[ testImage0Height ][ testImage0Width ][ OpenGlImage::pixelElementCount ] = 
    { { {  42,  1,  2,  3 }, {  4,  5,  6,  7 } },
    { {  8,  9, 10, 11 }, { 12, 13, 255, 15 } },
    { { 16, 17, 18, 19 }, { 20, 21, 22, 0 } }, };
    
    
    OpenSteer::size_t const testImage1Width = 2;
    OpenSteer::size_t const testImage1Height = 3;
    OpenGlImage::pixel_color_element_type imageData1[ testImage1Height ][ testImage1Width ][ OpenGlImage::pixelElementCount ] = 
    { { {  0,  1,  2,  3 }, {  4,  5,  6,  7 } }, // 0 different than color element form imageData0
    { {  8,  9, 10, 11 }, { 12, 13, 14, 15 } }, // 14 different than color element from imageData0
    { { 16, 17, 18, 19 }, { 20, 21, 22, 23 } }, }; // last color element different from imageData0
    
    
    OpenSteer::size_t const testImage2Width = 3;
    OpenSteer::size_t const testImage2Height = 2;
    OpenGlImage::pixel_color_type imageData2[ testImage2Height ][ testImage2Width ] =
    { { 1, 2, 3 },
      { 4, 5, 6 }, };
    
    
    OpenSteer::size_t const testImage3Width = 3;
    OpenSteer::size_t const testImage3Height = 2;
    OpenGlImage::pixel_color_element_type imageData3[ testImage3Height ][ testImage3Width ][ OpenGlImage::pixelElementCount ] =
        { { { 0, 0, 0, 1 }, { 0, 0, 0, 2 }, { 0, 0, 0, 3 } },
          { { 0, 0, 0, 4 }, { 0, 0, 0, 5 }, { 0, 0, 0, 6 } }, };   
    
    
    
    OpenSteer::size_t const testImage4Width = 3;
    OpenSteer::size_t const testImage4Height = 2;
    OpenGlImage::pixel_color_element_type imageData4[ testImage4Height ][ testImage4Width ][ OpenGlImage::pixelElementCount ] =
    { { { 0, 0, 0, 5 }, { 0, 0, 0, 5 }, { 0, 0, 0, 5 } },
    { { 0, 0, 0, 5 }, { 0, 0, 0, 5 }, { 0, 0, 0, 5 } }, };  
    
    
    OpenSteer::size_t const testImage5Width = 3;
    OpenSteer::size_t const testImage5Height = 2;
    OpenGlImage::pixel_color_element_type imageData5[ testImage5Height ][ testImage5Width ][ OpenGlImage::pixelElementCount ] =
    { { { 255, 255, 255, 255 }, { 255, 255, 255, 255 }, { 255, 255, 255, 255 } },
    { { 255, 255, 255, 255 }, { 255, 255, 255, 255 }, { 255, 255, 255, 255 } }, };  
    
} // namespace anonymous





OpenSteer::Graphics::OpenGlImageTest::OpenGlImageTest()
{
    // Nothing to do.
}



OpenSteer::Graphics::OpenGlImageTest::~OpenGlImageTest()
{
    // Nothing to do.
}




void 
OpenSteer::Graphics::OpenGlImageTest::setUp()
{
    TestFixture::setUp();
   
    testImage0.assign( testImage0Width, testImage0Height, &imageData0[ 0 ][ 0 ][ 0 ] );

    testImage1.assign( testImage1Width, testImage1Height, &imageData1[ 0 ][ 0 ][ 0 ] );
        
}



void 
OpenSteer::Graphics::OpenGlImageTest::tearDown()
{
    TestFixture::tearDown();
    
    testImage0.clear();
    testImage1.clear();
}





void 
OpenSteer::Graphics::OpenGlImageTest::testConstruction()
{
    OpenGlImage img0;
    CPPUNIT_ASSERT( 1 == img0.width() );
    CPPUNIT_ASSERT( 1 == img0.height() );
    CPPUNIT_ASSERT( 1 == img0.pixelCount() );
    CPPUNIT_ASSERT_EQUAL( std::numeric_limits< OpenGlImage::pixel_color_type >::max(), img0.pixel( 0, 0 ) );
    
    
    size_t width = 64;
    size_t height = 32;
    size_t pixelCount = width * height;
    OpenGlImage img1( width, height );
    CPPUNIT_ASSERT( width == img1.width() );
    CPPUNIT_ASSERT( height == img1.height() );
    CPPUNIT_ASSERT( pixelCount == img1.pixelCount() );
    
    
    width = 64;
    height = 32;
    pixelCount = width * height;
    OpenGlImage::pixel_color_type color( 0 );
    OpenGlImage img2( width, height, color );
    CPPUNIT_ASSERT( width == img2.width() );
    CPPUNIT_ASSERT( height == img2.height() );
    CPPUNIT_ASSERT( pixelCount == img2.pixelCount() ); 
    CPPUNIT_ASSERT( color == img2.pixel( 0, 0 ) );
    
    
    width = 64;
    height = 32;
    pixelCount = width * height;
    OpenGlImage::pixel_color_element_type r = 1; // left most byte
    OpenGlImage::pixel_color_element_type g = 2;
    OpenGlImage::pixel_color_element_type b = 3;
    OpenGlImage::pixel_color_element_type a = 4; // right most byte
    color = 16909060;
    OpenGlImage img3( width, height, r, g, b, a );
    CPPUNIT_ASSERT( width == img3.width() );
    CPPUNIT_ASSERT( height == img3.height() );
    CPPUNIT_ASSERT( pixelCount == img3.pixelCount() ); 
    CPPUNIT_ASSERT( color == img3.pixel( 0, 0 ) );
    
    
    size_t const width4 = 2;
    size_t const height4 = 3;
    size_t const pixelCount4 = width4 * height4;
    OpenGlImage::pixel_color_element_type testImage4[ height4 ][ width4 ][ OpenGlImage::pixelElementCount ] = 
        { { {  0,  1,  2,  3 }, {  4,  5,  6,  7 } },
          { {  8,  9, 10, 11 }, { 12, 13, 14, 15 } },
          { { 16, 17, 18, 19 }, { 20, 21, 22, 23 } }, };    
    OpenGlImage img4( width4, height4, &testImage4[ 0 ][ 0 ][ 0 ] );
    CPPUNIT_ASSERT_EQUAL( width4, img4.width() );
    CPPUNIT_ASSERT_EQUAL( height4, img4.height() );
    CPPUNIT_ASSERT_EQUAL( pixelCount4, img4.pixelCount() );
    for ( size_t i = 0; i != height4; ++i ) {
        for ( size_t j = 0; j != width4; ++j ) {
            for ( size_t k = 0; k != OpenGlImage::pixelElementCount; ++k ) {
                OpenGlImage::pixel_color_element_type original = testImage4[ i ][ j ][ k ];
                OpenGlImage::pixel_color_element_type copy = img4.pixelElement( j, i, k );
                // std::cout << "original " << static_cast< size_t >( original ) << " copy " << static_cast< size_t >( copy ) << " ";
                CPPUNIT_ASSERT_EQUAL( original, copy );
            }
            // std::cout << " | ";
        }
        // std::cout << std::endl;
    }
    
    
    size_t const width5 = 2;
    size_t const height5 = 3;
    size_t const pixelCount5 = width5 * height5;
    OpenGlImage::pixel_color_type testImage5[ height5 ][ width5 ] = 
    { { 0, 1 },
      { 2, 3 },
      { 4, 5 }, };    
    OpenGlImage img5( width5, height5, &testImage5[ 0 ][ 0 ] );
    CPPUNIT_ASSERT_EQUAL( width5, img5.width() );
    CPPUNIT_ASSERT_EQUAL( height5, img5.height() );
    CPPUNIT_ASSERT_EQUAL( pixelCount5, img5.pixelCount() );
    for ( size_t i = 0; i != height5; ++i ) {
        for ( size_t j = 0; j != width5; ++j ) {
            OpenGlImage::pixel_color_type original = testImage5[ i ][ j ];
            OpenGlImage::pixel_color_type copy = img5.pixel( j, i );
            // std::cout << "original " << static_cast< size_t >( original ) << " copy " << static_cast< size_t >( copy ) << " ";
            CPPUNIT_ASSERT_EQUAL( original, copy );
        }
        // std::cout << std::endl;
    }
    
    
    
    OpenGlImage img6( img4 );
    CPPUNIT_ASSERT_EQUAL( img4.width(), img6.width() );
    CPPUNIT_ASSERT_EQUAL( img4.height(), img6.height() );
    CPPUNIT_ASSERT_EQUAL( img4.pixelCount(), img6.pixelCount() );
    for ( size_t i = 0; i != img4.width(); ++i ) {
        for ( size_t j = 0; j != img4.height(); ++j ) {
            CPPUNIT_ASSERT_EQUAL( img4.pixel( i, j), img6.pixel( i, j ) );
            CPPUNIT_ASSERT_EQUAL( img4.pixelElement( i, j, OpenGlImage::R ), img6.pixelElement( i, j, OpenGlImage::R ) );
            CPPUNIT_ASSERT_EQUAL( img4.pixelElement( i, j, OpenGlImage::G ), img6.pixelElement( i, j, OpenGlImage::G ) );
            CPPUNIT_ASSERT_EQUAL( img4.pixelElement( i, j, OpenGlImage::B ), img6.pixelElement( i, j, OpenGlImage::B ) );
            CPPUNIT_ASSERT_EQUAL( img4.pixelElement( i, j, OpenGlImage::A ), img6.pixelElement( i, j, OpenGlImage::A ) );
        }
    }
    
    
    
}



void 
OpenSteer::Graphics::OpenGlImageTest::testComparison()
{
    
    OpenGlImage img0;
    OpenGlImage img1;    
    CPPUNIT_ASSERT( img0 == img1 );
    CPPUNIT_ASSERT( ! ( img0 != img1 ) ) ;
    
    CPPUNIT_ASSERT( testImage0 != testImage1 );
    CPPUNIT_ASSERT( testImage0 == testImage0 );
    CPPUNIT_ASSERT( testImage1 == testImage1 );
    
    size_t width2 = 1;
    size_t height2 = 2;
    OpenGlImage img2( width2, height2 );
    CPPUNIT_ASSERT( img2 != testImage0 );
    
}



void 
OpenSteer::Graphics::OpenGlImageTest::testAssignment()
{
    OpenGlImage img0;
    CPPUNIT_ASSERT( testImage0 != img0 );
    
    img0 = testImage0;
    CPPUNIT_ASSERT( testImage0 == img0 );
    
    img0 = testImage1;
    CPPUNIT_ASSERT( testImage0 != img0  );
    CPPUNIT_ASSERT( testImage1 == img0 );
    
    OpenGlImage img1( testImage2Width, testImage2Height, &imageData2[ 0 ][ 0 ] );
    OpenGlImage img2( testImage3Width, testImage3Height, &imageData3[ 0 ][ 0 ][ 0 ] );
    CPPUNIT_ASSERT( img1 == img2 );
    
    OpenGlImage img3;
    OpenGlImage img4;
    
    img3.assign( testImage2Width, testImage2Height, &imageData2[ 0 ][ 0 ] );
    img4.assign( testImage3Width, testImage3Height, &imageData3[ 0 ][ 0 ][ 0 ] );
    CPPUNIT_ASSERT( img3 == img4 );
    
    img3 = testImage0;
    CPPUNIT_ASSERT( img3 != img4 );
    
}





void 
OpenSteer::Graphics::OpenGlImageTest::testSwap()
{
    OpenGlImage img0;
    OpenGlImage img1( testImage0 );
    CPPUNIT_ASSERT( img0 != img1 );
    
    swap( img0, img1 );
    CPPUNIT_ASSERT( img0 != img1 );
    CPPUNIT_ASSERT( img0 == testImage0 );
    CPPUNIT_ASSERT( img1 == OpenGlImage() );
    
}



void
OpenSteer::Graphics::OpenGlImageTest::testSettingPixels() 
{
    OpenGlImage img0( testImage0 );
    OpenGlImage img1( testImage1 );
    CPPUNIT_ASSERT( img0 != img1 );
    
    CPPUNIT_ASSERT( 42 == img0.pixelElement( 0, 0, OpenGlImage::R ) );
    img0.setPixel( 0, 0, 66051 );
    CPPUNIT_ASSERT( 0 == img0.pixelElement( 0, 0, OpenGlImage::R ) );    
    CPPUNIT_ASSERT( img0 != img1 );
    CPPUNIT_ASSERT( 255 == img0.pixelElement( 1, 1, OpenGlImage::B ) );
    img0.setPixel( 1, 1, 12, 13, 14, 15 );
    CPPUNIT_ASSERT( 14 == img0.pixelElement( 1, 1, OpenGlImage::B ) );
    CPPUNIT_ASSERT( img0 != img1 );
    CPPUNIT_ASSERT( 0 == img0.pixelElement( 1, 2, OpenGlImage::A ) );
    img0.setPixelElement( 1, 2, OpenGlImage::A, 23 );
    CPPUNIT_ASSERT( 23 == img0.pixelElement( 1, 2, OpenGlImage::A ) );
    CPPUNIT_ASSERT( img0 == img1 );
    
}



void
OpenSteer::Graphics::OpenGlImageTest::testClear() 
{
    OpenGlImage referenceImage( testImage4Width, testImage4Height, &imageData4[ 0 ][ 0 ][ 0 ] );
    OpenGlImage img0( testImage3Width, testImage3Height, &imageData3[ 0 ][ 0 ][ 0 ] );
    
    CPPUNIT_ASSERT( img0 != referenceImage );
    img0.clear( 0, 0, 0, 5 );
    CPPUNIT_ASSERT( img0 == referenceImage );
    
    img0.assign( testImage3Width, testImage3Height, &imageData3[ 0 ][ 0 ][ 0 ] );
    CPPUNIT_ASSERT( img0 != referenceImage );
    img0.clear( 5 );
    CPPUNIT_ASSERT( img0 == referenceImage );
    
    
    referenceImage.assign( testImage5Width, testImage5Height, &imageData5[ 0 ][ 0 ][ 0 ] );
    CPPUNIT_ASSERT( img0 != referenceImage );
    img0.clear();
    CPPUNIT_ASSERT( img0 == referenceImage );
    
    
    
}


void
OpenSteer::Graphics::OpenGlImageTest::testCreatePixelColorElementFromAFloat()
{
    CPPUNIT_ASSERT_EQUAL( 255, static_cast< int >( std::numeric_limits< OpenGlImage::pixel_color_element_type >::max() ) );
    
    float const sourceColorElement0 = 0.0;
    OpenGlImage::pixel_color_element_type targetColorElement0 = makePixelColorElement( sourceColorElement0 );
    CPPUNIT_ASSERT_EQUAL( 0, static_cast< int >( targetColorElement0 ) );
    
    float const sourceColorElement1 = 1.0;
    OpenGlImage::pixel_color_element_type targetColorElement1 = makePixelColorElement( sourceColorElement1 );
    CPPUNIT_ASSERT_EQUAL( 255, static_cast< int >( targetColorElement1 ) );
    
    
}




