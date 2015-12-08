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
// ProximityMap
//
// This is a generic list to store pairs of references and values.
//
// 22-06-06 jb:  created
//
//
// ----------------------------------------------------------------------------


#ifndef INCLUDED_OpenSteer_ProximityMap_H
#define INCLUDED_OpenSteer_ProximityMap_H

/** @author Jens Breitbart <http://www.jensmans-welt.de/contact> */

#include <map>
#include <utility>
#include <functional>
#include <algorithm>
#include <cmath>
#include "OpenSteer/Vec3.h"

namespace OpenSteer {
/**
 * requirements: - unique references
 *               - referenceT::operator<
 */

template <typename referenceT, typename dataT=Vec3>
class ProximityList {
	private:
		typedef std::map<referenceT, dataT> datastructure_type;

	public:
		/*** iterators typedefs ***/
		typedef typename datastructure_type::const_iterator const_iterator;
		typedef typename datastructure_type::iterator iterator;

		/*** constructors & destructors ***/
		ProximityList() {};
		ProximityList(const ProximityList<referenceT, dataT>& copy) : datastructure_(copy) {};

		ProximityList<referenceT, dataT> operator= (const ProximityList<referenceT, dataT> &rhs) {
			if (&rhs==this) return *this;
			datastructure_=rhs.datastructure_;
			return *this;
		}

		~ProximityList() {};

		/*** functions for our clients ***/

		/**
		 * adds a (refernceT, dataT) pair to our datastructure
		 * @param ref
		 * @param position
		 */
		void add( const referenceT& ref, const dataT& position );

		/**
		 * removes the pair referred by ref from our datastructure
		 * @param ref
		 */
		void remove( const referenceT& ref );

		/**
		 * removes the pair referred by ref from our datastructure, may be faster than
		 * <code>void remove( const referenceT& ref )</code>
		 * @param ref
		 * @param position
		 */
		void remove( const referenceT& ref, const dataT& position );

		/**
		 * updates the dataT value from the pair referred by ref
		 * @param ref
		 * @param new_position
		 */
		void update( const referenceT& ref, const dataT& new_position );

		/**
		 * updates the dataT value from the pair referred by ref, may be faster than,
		 * <code>void update( const referenceT& ref, const dataT& new_position )</code>
		 * @param ref
		 * @param new_position
		 * @param old_position
		 */
		void update( const referenceT& ref, const dataT& new_position, const dataT& old_position );

		/**
		 * function to find all neighbours
		 * @param position
		 * @param max_radius
		 * @param iter
		 */
		template< typename OutPutIterator >
		void find_neighbours( const dataT &position, const float max_radius, OutPutIterator &iter ) const;

		/*** usefull iterator functions ***/
		const_iterator begin() const;
		const_iterator end() const;
		iterator begin();
		iterator end();



//really usefull???
		// Assumes a <code> DataT position() const</code> member function.
/*		void add( ProximityData const& data ); // proxy.add( ProximityData< RefType, Vec3>( ref, position() ) ); // add( make_proximity_data()  ref, position() );
		void remove( ProximityData const&  data );
		void find_neighbors( ProximityData data, max_radius, ProximityList<> returnee ) const;
*/


	private:
		/**
		 * two find functions, just wrappers around std::find
		 */
		const_iterator find(const referenceT &ref, const dataT &position) const {
			return datastructure_.find(ref);
		}
		iterator find(const referenceT &ref, const dataT &position) {
			return datastructure_.find(ref);
		}

	private:
		/**
		 * the heart of our list ... the datastructure
		 */
		datastructure_type datastructure_;
};



template <typename referenceT, typename dataT>
void ProximityList<referenceT, dataT>::add( const referenceT& ref, const dataT& position ) {
	datastructure_.insert(std::make_pair(ref, position));
}


template <typename referenceT, typename dataT>
void ProximityList<referenceT, dataT>::remove( const referenceT& ref, const dataT& position ) {
	datastructure_.erase(ref);
}


template <typename referenceT, typename dataT>
void ProximityList<referenceT, dataT>::remove( const referenceT& ref ) {
	datastructure_.erase(ref);
}


template <typename referenceT, typename dataT>
void ProximityList<referenceT, dataT>::update( const referenceT& ref, const dataT& new_position, const dataT& old_position ) {
	datastructure_.insert(std::make_pair(ref, new_position));
}


template <typename referenceT, typename dataT>
void ProximityList<referenceT, dataT>::update( const referenceT& ref, const dataT& new_position ) {
	datastructure_.insert(std::make_pair(ref, new_position));
}


template< typename referenceT, typename dataT >
template< typename OutPutIterator >
void ProximityList<referenceT, dataT>::find_neighbours( const dataT &position, const float max_radius, OutPutIterator &iter ) const {
	for (const_iterator i=datastructure_.begin(); i!=datastructure_.end(); ++i) {
		const float temp=abs(i->second-position);
		if (temp<max_radius) {
			*iter = i->first;
			++iter;
		}
	}
}


template <typename referenceT, typename dataT>
typename ProximityList<referenceT, dataT>::const_iterator ProximityList<referenceT, dataT>::begin() const {
	return datastructure_.begin();
}


template <typename referenceT, typename dataT>
typename ProximityList<referenceT, dataT>::const_iterator ProximityList<referenceT, dataT>::end() const {
	return datastructure_.end();
}


template <typename referenceT, typename dataT>
typename ProximityList<referenceT, dataT>::iterator ProximityList<referenceT, dataT>::begin() {
	return datastructure_.begin();
}


template <typename referenceT, typename dataT>
typename ProximityList<referenceT, dataT>::iterator ProximityList<referenceT, dataT>::end() {
	return datastructure_.end();
}


}

#endif // INCLUDED_OpenSteer_ProximityMap_H

