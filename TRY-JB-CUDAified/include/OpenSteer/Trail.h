#ifndef OPENSTEER_TRAIL_H
#define OPENSTEER_TRAIL_H


// Include assert
#include <cassert>

// Include std::vector
#include <vector>

// Include std::swap
#include <algorithm>



// Include OpenSteer::Vec3
#include "OpenSteer/Vec3.h"

// Include OpenSteer::size_t
#include "OpenSteer/StandardTypes.h"

// Include OPENSTEER_UNUSED_PARAMETER
#include "OpenSteer/UnusedParameter.h"


namespace OpenSteer {
    
    /**
     * Represents the trail a vehicle travelled inside a given duration. Kind of
     * like the footsteps along the way.
     *
     * First all footsteps are set at the zero position therefore always all
     * possible footsteps are stored even if not enough footstep positions 
     * have been recorded.
     *
     * Automatically compiled standard copy-constructor and assignment-operator
     * are working as expected and are therefore not written explicitly.
     *
     * @attention @c FootstepCount must be at least @c 1 before querying any
     *            position, footstep, or time data.
     */
    template< size_t FootstepCount >
    class Trail {
    public:
        typedef size_t size_type;
        
        /**
         * Creates a trail for a trail duration of 5 seconds.
         */
        Trail() 
        : footstepPositions_( FootstepCount * 2), 
            footstepPositionTimes_( FootstepCount * 2 ), 
            footstepCount_( 0 ), 
            nextFootstepIndex_( 0 ), 
            reciprocalDuration_( 1.0f / 5.0f ), 
            lastPosition_( 0.0f, 0.0f, 0.0f ), 
            lastPositionTime_( 0.0f ), 
            footDown_( false ) {
            // Nothing to do.
        }
        
        /**
         * Creates a trail for a trail duration of @a _duration.
         *
         * A float value of @c 1.0f represents one second.
         *
         * @attention _duration must be greater than @c 0.0fs.
         */
        explicit Trail( float _duration )
            : footstepPositions_( FootstepCount * 2), 
            footstepPositionTimes_( FootstepCount * 2 ), 
            footstepCount_( 0 ), 
            nextFootstepIndex_( 0 ), 
            reciprocalDuration_( 1.0f / _duration ), 
            lastPosition_( 0.0f, 0.0f, 0.0f ), 
            lastPositionTime_( 0.0f ), 
            footDown_( false ) {
            // Nothing to do.
        }
        
        
        void swap( Trail& _other ) {
            footstepPositions_.swap( _other.footstepPositions_ );
            footstepPositionTimes_.swap( _other.footstepPositionTimes_ );
            std::swap( footstepCount_, _other.footstepCount_ );
            std::swap( nextFootstepIndex_, _other.nextFootstepIndex_ );
            std::swap( reciprocalDuration_, _other.reciprocalDuration_ );
            lastPosition_.swap( _other.lastPosition_ );
            std::swap( lastPositionTime_, _other.lastPositionTime_ );
            std::swap( footDown_, _other.footDown_ );
        }
        
        
        // @todo Just added for performance observations.
        typedef typename std::vector< Vec3 >::const_iterator const_footstep_position_iterator;
        // @todo Just added for performance observations.
        typedef typename std::vector< float >::const_iterator const_footstep_position_time_iterator;
    
        /**
         * @todo This needs proper testing and is just inserted for some
         *       performance observations.
         */
        const_footstep_position_iterator beginFootstepPositions() const {
            return footstepPositions_.begin();
        }
        
        /**
         * @todo This needs proper testing and is just inserted for some
         *       performance observations.
         */
        const_footstep_position_iterator endFootstepPositions() const {
            return footstepPositions_.end();
        }
        
        /**
         * @todo This needs proper testing and is just inserted for some
         *       performance observations.
         */
        const_footstep_position_time_iterator beginFootstepPositionTimes() const {
            return footstepPositionTimes_.begin();
        }
        
        /**
         * @todo This needs proper testing and is just inserted for some
         *       performance observations.
         */
        const_footstep_position_time_iterator endFootstepPositionTimes() const {
            return footstepPositionTimes_.end();
        }
        
        
        
        
        
        
        /**
         * If no new footsteps are produced but time goes by it cuts-of all 
         * footsteps that are too far back in time.
         *
         * @todo Implement. However as this functionality isn't contained in
         *       the current incarnation of OpenSteer trails it isn't 
         *       implemented either.
         */
        void tickTime( float _time ) {
			OPENSTEER_UNUSED_PARAMETER( _time );
            
            // @todo Don't forget to set the last stored position when footDown_
            //       became @c true to this time.
        }
        
        
        /**
         * Two consecutive positons build one step. The next new consecutive
         * positions form the next step and so on.
         *
         * Each newly inserted position time must be later or higher than the
         * last recorded position time.
         */
        void recordPosition( Vec3 const& _position, float _currentTime ) {
            assert( ( _currentTime >= lastPositionTime_ ) && 
                    "_currentTime of _position is lesser than the last recorded"
                    " time. This can't be in a well formed trail." );
            
            if ( ! footDown_ ) {
                // A new footstep starts (the heel touches the ground).
                footDown_ = true;
                lastPosition_ = _position;
                lastPositionTime_ = _currentTime;
            } else {
                // The toes leave the ground therefore the last footstep has
                // been completed.
                size_type const footstepPositionStartIndex = nextFootstepIndex_ * 2;
                
                footstepPositions_[ footstepPositionStartIndex ] = lastPosition_;
                footstepPositionTimes_[ footstepPositionStartIndex ] = lastPositionTime_;
                
                footDown_ = false;
                lastPosition_ = _position;
                lastPositionTime_ = _currentTime;
                
                footstepPositions_[ footstepPositionStartIndex + 1 ] = _position;
                footstepPositionTimes_[ footstepPositionStartIndex + 1 ] = _currentTime;
                
                // Increase the footstep count if the storage ring isn't full.
                if ( maxFootstepCount() > footstepCount() ) {
                    ++footstepCount_;
                }
                
                // Increase the index where to store the next footstep. Use 
                // module because of the ring structure of the storage.
                ++nextFootstepIndex_;
                nextFootstepIndex_ %= maxFootstepCount();
                
                // Prune old data.
                tickTime( _currentTime );
            }
            
            
            
        }
        
        
        void clear() {
            footstepPositions_.clear();
            footstepPositionTimes_.clear();
            footstepCount_ = 0;
            nextFootstepIndex_ = 0;
            lastPosition_ = Vec3( 0.0f, 0.0f, 0.0f );
            lastPositionTime_ = 0.0f;
            footDown_ = false;
        }
        
        /**
         * Returns how many complete footsteps are stored.
         */
        size_type footstepCount() const {
            return footstepCount_;
        }
        
        /**
         * Returns the number of positions that form footsteps. If a position
         * has been recorded that is missing a consecutive positon to build a 
         * footstep it isn't counted.
         */
        size_type positionCount() const {
            return footstepCount() * 2;
        }
        
        
        size_type maxFootstepCount() const {
            return FootstepCount;
        }

        size_type maxPositionCount() const {
            return maxFootstepCount() * 2;
        }
        
        bool empty() const {
            return 0 == footstepCount_;
        }
        
        bool full() const {
            return footstepCount() == maxFootstepCount();
        }
        
        /**
         * Returns the footstep position that is currently indexed by @a _index.
         * There are twice as many positions than footsteps stored.
         *
         * @attention The footstep stored at index @c 0 isn't the currently
         *            first footstep stored in time but first stored in
         *            continuous memory. To get access to the first footstep in
         *            time use @c footstepPositionAtTime.
         *
         * @attention Only call if at least one footstep is contained.
         */
        Vec3 const& footstepPosition( size_type _index ) const {
            assert( ( _index < ( footstepCount() * 2 ) ) && "_index out of range." );
            return footstepPositions_[ _index ];
        }
        
        /**
         * Returns the footstep position stored at @a _index, and index of @c 0
         * references the first position stored in time.
         *
         * @attention Only call if at least one footstep is contained.
         */
        Vec3 const& footstepPositionAtTime( size_type _index ) const {
            return footstepPosition( positionIndexAtTime( _index ) );
        }
        
        
        
        /**
         * @attention Only call if at least one footstep is contained.
         */
        /*
        float const& footstepPositionData() const {
            return footstepPositions_[ 0 ].data();
        }
        */
        
        /**
         * Returns @c true if the footstep indexed by @a _index has been stored
         * at a second boundary, @c false otherwise.
         */
        bool footstepAtTick( size_type _index, float _tickDuration ) const {
            assert( _index < footstepCount() && "_index out of range." );
            
            size_type predecessingIndex = predecessingFootstepIndex( _index );
            
            if ( ( _tickDuration <= ( footstepPositionTime( ( _index * 2 ) + 1) - 
                                    footstepPositionTime( _index * 2 ) ) ) || 
                 ( _tickDuration <= ( footstepPositionTime( _index * 2 ) - 
                                      footstepPositionTime( ( predecessingIndex * 2 ) + 1) ) ) ) {
                // The footstep took longer than @c _tickDuration from touching
                // the ground until leaving the ground or from leaving the
                // ground before the footstep at @c _index until touching the 
                // ground with the footstep at @c _index.
                return true;
            } 
            
            return false;
        }
        
        
        bool footstepAtTickAtTime( size_type _index, float _tickDuration ) const {
            return footstepAtTick( ( nextFootstepIndex_ + _index ) % footstepCount(), 
                                   _tickDuration );
        }
        
        
        float footstepPositionTime( size_type _index ) const {
            assert( ( _index < ( footstepCount() * 2 ) ) && "_index out of range." );
            return footstepPositionTimes_[ _index ];
        }
        
        float footstepPositionTimeAtTime( size_type _index ) const {
            return footstepPositionTime( positionIndexAtTime( _index ) );
        }
        
        
        float positionSampleInterval() const {
            return footstepCount() * reciprocalDuration_;
        }
            
        float duration() const {
            return 1.0f / reciprocalDuration_;
        }
        
        void setDuration( float _duration ) {
            assert( _duration > 0.0f && "_duration must be greater than zero." );
            reciprocalDuration_ = 1.0f / _duration;
            
            // Cut-off all footstep data that is to far back in time.
            tickTime( lastPositionTime_ );
        }
        
        Vec3 const& lastFootstepPosition() const {
            return lastPosition_;
        }
        
        
        float lastFootstepPositionTime() const {
            return lastPositionTime_;
        }
        
        
    private:
            
        size_type positionIndexAtTime( size_type _index ) const {
            return ( ( ( nextFootstepIndex_ * 2 ) + _index ) % ( footstepCount() * 2 ) );
        }
        
        /**
         * Returns the index of the footstep before @a _index. If no earlier 
         * footstep than the one at @a _index exists @a _index is returned.
         *
         * Don't call if no footsteps are contained in the trail.
         */
        size_type predecessingFootstepIndex( size_type _index ) const {
            
            size_type result = _index;
            
            if ( _index == nextFootstepIndex_ ) {
                return result;
            }
            
            if ( 0 != _index ) {
                --result;
            } else {
                result = footstepCount() - 1;
            }
            
            return result;
        }
        
        
        
    private:
        // A ring for storing footsteps. Two Vec3s define one step. There should
        // never be an odd number of Vec3s in the ring.
        std::vector< Vec3 > footstepPositions_;
        
        // Time of footstep position recording.
        std::vector< float > footstepPositionTimes_;
            
        // How many footsteps have been stored.
        size_type footstepCount_;
        
        // Index where to add footsteps to the ring @c footsteps_.
        size_type nextFootstepIndex_;
        
        float reciprocalDuration_;
    
        Vec3 lastPosition_;
        float lastPositionTime_;
        
        // Marks the beginning of a footstep. If a footstep is added and it is
        // @c true then a complete footstep can be added to @c footsteps_,
        // otherwise the footstep just begins.
        bool footDown_;
        
    }; // class Trail
    
    
    /**
     * Not implemented specialization to prevent a footstep count of @c 0.
     */
    template<> class Trail< 0 >;
    
    
    template< size_t FootstepCount >
        void swap( Trail< FootstepCount >& lhs, Trail< FootstepCount >& rhs ) {
            lhs.swap( rhs );
        }
    
    
    
    
    
} // namespace OpenSteer

#endif // OPENSTEER_TRAIL_H

