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

// @todo Rename to proximity_associated_container.

#ifndef INCLUDED_OpenSteer_proximity_associated_container_H
#define INCLUDED_OpenSteer_proximity_associated_container_H

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
class proximity_associated_container {
	private:
		typedef std::map<referenceT, dataT> datastructure_type;

	public:
		/*** iterators typedefs ***/
		typedef typename datastructure_type::const_iterator const_iterator;
		typedef typename datastructure_type::iterator iterator;

		/*** constructors & destructors ***/
		proximity_associated_container() {};
		proximity_associated_container(const proximity_associated_container<referenceT, dataT>& copy) : datastructure_(copy) {};

		proximity_associated_container<referenceT, dataT> operator= (const proximity_associated_container<referenceT, dataT> &rhs) {
			if (&rhs==this) return *this;
			datastructure_=rhs.datastructure_;
			return *this;
		}

		~proximity_associated_container() {};

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
void proximity_associated_container<referenceT, dataT>::add( const referenceT& ref, const dataT& position ) {
	datastructure_.insert(std::make_pair(ref, position));
}


template <typename referenceT, typename dataT>
void proximity_associated_container<referenceT, dataT>::remove( const referenceT& ref, const dataT& position ) {
	datastructure_.erase(ref);
}


template <typename referenceT, typename dataT>
void proximity_associated_container<referenceT, dataT>::remove( const referenceT& ref ) {
	datastructure_.erase(ref);
}


template <typename referenceT, typename dataT>
void proximity_associated_container<referenceT, dataT>::update( const referenceT& ref, const dataT& new_position, const dataT& old_position ) {
	datastructure_.insert(std::make_pair(ref, new_position));
}


template <typename referenceT, typename dataT>
void proximity_associated_container<referenceT, dataT>::update( const referenceT& ref, const dataT& new_position ) {
	datastructure_.insert(std::make_pair(ref, new_position));
}


template< typename referenceT, typename dataT >
template< typename OutPutIterator >
void proximity_associated_container<referenceT, dataT>::find_neighbours( const dataT &position, const float max_radius, OutPutIterator &iter ) const {
	for (const_iterator i=datastructure_.begin(); i!=datastructure_.end(); ++i) {
		const float temp=abs(i->second-position);
		if (temp<max_radius) {
			*iter = i->first;
			++iter;
		}
	}
}


template <typename referenceT, typename dataT>
typename proximity_associated_container<referenceT, dataT>::const_iterator proximity_associated_container<referenceT, dataT>::begin() const {
	return datastructure_.begin();
}


template <typename referenceT, typename dataT>
typename proximity_associated_container<referenceT, dataT>::const_iterator proximity_associated_container<referenceT, dataT>::end() const {
	return datastructure_.end();
}


template <typename referenceT, typename dataT>
typename proximity_associated_container<referenceT, dataT>::iterator proximity_associated_container<referenceT, dataT>::begin() {
	return datastructure_.begin();
}


template <typename referenceT, typename dataT>
typename proximity_associated_container<referenceT, dataT>::iterator proximity_associated_container<referenceT, dataT>::end() {
	return datastructure_.end();
}


}

#endif // INCLUDED_OpenSteer_proximity_associated_container_H

