#ifndef OPENSTEER_PLUGINUTILITIES_H
#define OPENSTEER_PLUGINUTILITIES_H

// Include std::back_insert_iterator
#include <iterator>

// Include std::binary_function
#include <functional>

// Include std::priority_qeue
#include <queue>

// Include AVGroup
#include "OpenSteer/AbstractVehicle.h"

// Include Opensteer::ProximitList
#include "OpenSteer/ProximityList.h"

// Include Opensteer::Vec3
#include "OpenSteer/Vec3.h"


namespace OpenSteer {
	
	/**
	 * Compares the distance of two vehicles to a given position.
	 *
	 * @todo Write unit test.
	 */
	class compare_distance : public std::binary_function< OpenSteer::AbstractVehicle const*, OpenSteer::AbstractVehicle const*, bool > {
	public:
		
		compare_distance( OpenSteer::Vec3 const& distance_to )
		: distance_to_( distance_to ) {
			// Nothing to do.
		}
		
		/**
		 * @return @c true if @a rhs is nearer to a given point than @c lhs, @c false 
		 *         otherwise.
		 */
		bool operator()( OpenSteer::AbstractVehicle const* lhs, OpenSteer::AbstractVehicle const* rhs ) {
			OpenSteer::Vec3::value_type distance_lhs = ( distance_to_ - lhs->position() ).lengthSquared();
			OpenSteer::Vec3::value_type distance_rhs = ( distance_to_ - rhs->position() ).lengthSquared();
			
			// Return @c true if @a rhs is nearer to @c distance_to_ than @c lhs, @c false 
			// otherwise.
			return distance_lhs > distance_rhs;
		}
		
	private:
		OpenSteer::Vec3 const distance_to_;
		
	}; // class compare_distance
	
	
	
	
	/**
	 * Specialization of @c OpenSteer::ProximityList::find_neighbours to work with
	 * @c OpenSteer::AbstractVehicles.
	 */
	template< >
		template< typename OutPutIterator >
		void OpenSteer::ProximityList< AbstractVehicle* , OpenSteer::Vec3>::find_neighbours( const Vec3 &position,  float const max_radius, OutPutIterator iter ) const {
			// @todo Remove magic variable.
			std::size_t const neighbour_size_max = 7;
			std::size_t max_neighbour_distance_index = 0;
			float max_neighbour_distance = 0.0f;
			std::vector< AbstractVehicle* > neighbours;
			neighbours.reserve( neighbour_size_max );
			
			
			float const r2 = max_radius*max_radius;
			for (const_iterator i=datastructure_.begin(); i != datastructure_.end(); ++i) {
				Vec3 const offset = position - i->second;
				float const d2 = offset.lengthSquared();
				// Enabling the following two lines and disabling the current "if" triggers a message
				// in the Shark profiling tool that this function is auto-vectorizable.
				// int difference = static_cast< int >( r2 - d2 );
				// if ( difference > 0 ) {
				if (d2<r2) {
					// *iter = i->first;
					// ++iter;
					
					
					if ( neighbours.size() < neighbour_size_max ) {
						if ( d2 > max_neighbour_distance ) {
							max_neighbour_distance = d2;
							max_neighbour_distance_index = neighbours.size();
						}
						neighbours.push_back( i->first );
					} else {
						
						if ( d2 < max_neighbour_distance ) {
							neighbours[ max_neighbour_distance_index ] = i->first;
							max_neighbour_distance = d2; // just temporary
							
							for ( std::size_t i = 0; i < neighbour_size_max; ++i ) {
								
								float const dist = ( position - neighbours[ i ]->position() ).lengthSquared();
								if ( dist > max_neighbour_distance ) {
									max_neighbour_distance = dist;
									max_neighbour_distance_index = i;
								}
							}
						}
						
					}
					
				}
			}
			
			for ( std::size_t i = 0; i < neighbours.size(); ++i ) {
				*iter = neighbours[ i ];
				++iter;
			}
			
		} // OpenSteer::ProximityList< AbstractVehicle* , OpenSteer::Vec3>::find_neighbours
	
	
	
	
} // namespace OpenSteer


namespace std {

	/**
	 * Full specialization of @c std::back_insert_iterator to work with @c std::priority_queue.
	 *
	 * @todo Write unit test.
	 */
	template<>
	class back_insert_iterator<  std::priority_queue< OpenSteer::AbstractVehicle*, OpenSteer::AVGroup, OpenSteer::compare_distance > >: public std::iterator< std::output_iterator_tag, void, void, void, void > {
	public:
		typedef std::priority_queue< OpenSteer::AbstractVehicle*, OpenSteer::AVGroup, OpenSteer::compare_distance > container_type;
		
		explicit back_insert_iterator( container_type& x ) : container_( &x ) {}
		
		back_insert_iterator< container_type >& operator=( container_type::const_reference value ) {
			container_->push( value );
			
			return *this;
		}
		
		back_insert_iterator< container_type >& operator*() {
			return *this;
		}
		back_insert_iterator< container_type >& operator++() { return *this; }
		back_insert_iterator< container_type >& operator++(int) { return *this; }
		
	protected:
		container_type* container_;
		
	}; // std::back_insert_iterator

} // namespace std


#endif // OPENSTEER_PLUGINUTILITIES_H
