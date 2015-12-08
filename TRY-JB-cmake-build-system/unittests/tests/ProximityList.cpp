// ----------------------------------------------------------------------------
//
//
// OpenSteer -- Steering Behaviors for Autonomous Characters
//
// Copyright (c) 2002-2005, Sony Computer Entertainment America
// Original author: Craig Reynolds <craig_reynolds@playstation.sony.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//
// ----------------------------------------------------------------------------
//
//
// ProximityData Testcase
//
// A unittestcase for the class ProximityData
//
// 22-06-06 jb:  created
//
//
// ----------------------------------------------------------------------------

/** @author Jens Breitbart <http://www.jensmans-welt.de/contact> */

#include <vector>
#include <iterator>

#include <UnitTest++/UnitTest++.h>
// #include "../unittest++/UnitTest++.h"
// #include "../unittest++/TestReporter.h"
// #include "../unittest++/ReportAssert.h"
// #include "../unittest++/Config.h"
#include "OpenSteer/ProximityList.h"
#include "OpenSteer/Vec3.h"

namespace OpenSteer {
template< >
template< typename OutPutIterator >
void ProximityList<Vec3*, Vec3>::find_neighbours( const Vec3 &position, const float max_radius, OutPutIterator iter ) const {
	const double r2 = max_radius*max_radius;
	for (const_iterator i=datastructure_.begin(); i!=datastructure_.end(); ++i) {
		const Vec3 offset = position - i->second;
		const double d2 = offset.lengthSquared();
		if (d2<r2) {
			*iter = i->first;
			++iter;
		}
	}
}
}

namespace {

TEST(ProximityList_add_remove) {
	using namespace OpenSteer;
	ProximityList<int, double> list;
	list.add( 0, 0.0);
	list.add( 1, 1.0);
	list.remove( 0);
	list.remove( 1, 1.0);

	//check if list is empty
 	CHECK(list.begin()==list.end());
}

TEST(ProximityList_add_update_iterate) {
	using namespace OpenSteer;
	ProximityList<int, double> list;
	list.add( 0, 0.0);
	list.add( 1, 1.0);
	list.update( 0, 23.0);
	list.update( 1, 42.0);

	bool test=true;

	for (ProximityList<int, double>::const_iterator iter=list.begin(); iter!=list.end(); ++iter) {
		if (iter->first==0) {
			test=test&&(iter->second==23.0);
		}
		if (iter->first==1) {
			test=test&&(iter->second==42.0);
		}
	}

	CHECK(test);
}

TEST(ProximityList_Vec3_find_neighbours) {
	using namespace OpenSteer;
	using namespace std;
	ProximityList<Vec3*> list;
	Vec3 a(0, 0, 0);
	Vec3 b(10, 0, 0);
	Vec3 c(5, 0, 0);
	Vec3 d(5, 5, 5);
	list.add( &a, a);
	list.add( &b, b);
	list.add( &c, c);
	list.add( &d, d);

	vector<Vec3*> vecvec;

	back_insert_iterator< vector< Vec3* > > backi(vecvec);
	list.find_neighbours( Vec3(0,0,0), 6.0, backi);

	bool test=true;

	for (vector<Vec3*>::const_iterator iter=vecvec.begin(); iter!=vecvec.end(); ++iter) {
		if (*iter==&a || *iter==&c) {
		} else {
			test=false;
		}
	}

	CHECK(test);
}

}
