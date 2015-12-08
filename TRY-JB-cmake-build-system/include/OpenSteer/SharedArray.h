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
 * Reference counting smart pointer holding arrays.
 */
#ifndef OPENSTEER_SHAREDARRAY_H
#define OPENSTEER_SHAREDARRAY_H


// Include std::swap
#include <algorithm>

// Include assert
#include <cassert>




// Include OpenSteer::size_t
#include "OpenSteer/StandardTypes.h"



namespace OpenSteer {
    
    /**
    * Helper class for @c SharedPointer.
     */
    struct SharedArrayReferenceCount {
        typedef size_t size_type;
        
        
        explicit SharedArrayReferenceCount(): referenceCount_( 1 ) {
            // Nothing to do.
        }
        
        size_type referenceCount_;
    };
    
    
    /**
    * Smart pointer implementation using reference counting. Dereferencing is
     * as fast as it can get and might reach the speed of dereferencing raw
     * pointers if a good optimizing compiler is used.
     *
     * Doesn't manage arrays.
     *
     * Isn't designed to be used outside of OpenSteer. Boost library
     * smart pointers are preferable if a solid and platform agnostic 
     * implementation is needed. It should be quite easy to replace this smart
     * pointer with a Boost @c shared_ptr because of many intentional design 
     * parallels.
     *
     * See Andrei Alexandrescu, Modern C++ Design, Addison-Wesley, 2002, 
     * pp. 165--167.
     *
     * See http://Boost.org shared_ptr 
     * ( http://www.boost.org/libs/smart_ptr/shared_ptr.htm ).
         *
         * @attention Beware of cycles of smart pointers as these will lead to 
         * memory leaks.
         */
    template< typename T >
        class SharedArray {
        public:
            typedef size_t size_type;
            typedef T value_type;
            typedef value_type& reference;
            typedef value_type const& const_reference;
            typedef value_type* pointer;
            typedef value_type const* const_pointer;
            
            template< typename U > friend class SharedArray;
            
            
            /**
             * Constructs an empty @c SharedArray. The use count is unspecified.
             *
             * @post <code>get() == 0</code>
             *
             * @throw @c std::bad_alloc if memory could not be obtained.
             */
            SharedArray() : data_( 0 ), referenceCount_( new SharedArrayReferenceCount() ) {
                // Nothing to do.
            }
            
            /**
             * Constructs a @c SharedArray that owns the pointer @a _data.
             *
             * @post <code> use_count() == 1 </code> and <code>get() == _data )</code>
             *
             * @throw @c std::bad_alloc if memory could not be obtained.
             */
            explicit SharedArray( T* _data ) : data_( _data ), referenceCount_( new SharedArrayReferenceCount() ) {
                // Nothing to do.
            }
            
            /**
             * Constructs a @c SharedArray that shares ownership with @a other.
             *
             * @post <code>get() == other.get()</code> and 
             *       <code>useCount() == other.useCount()</code>
             *
             * @throw Nothing.
             */
            SharedArray( SharedArray const& other ) : data_( other.data_ ), referenceCount_( other.referenceCount_ ) {
                retain();
            }
            
            /**
             * Constructs a @c SharedArray that shares ownership with @a other.
             * A pointer of type @c U must be assignable to a pointer of type @c T.
             *
             * @post <code>get() == other.get()</code> and 
             *       <code>useCount() == other.useCount()</code>
             *
             * @throw Nothing. 
             */
            template< typename U >
                SharedArray( SharedArray< U > const& other ) : data_( other.data_ ), referenceCount_( other.referenceCount_ ) {
                    retain();
                }
            
            /**
             * Decreases the use count by one. If the use count hits @c 0 the 
             * managed pointer is deleted.
             *
             * @throw Might throw an exception if the destructor of the managed 
             *        pointer throws an exception if called.
             */
            ~SharedArray() {
                release();
            }
            
            /**
             * Shares the ownership of the pointer managed by @a other with @a other.
             * The old managed pointer is deleted if ownership decreases to @c 0.
             *
             * @throw Nothing.
             */
            SharedArray& operator=( SharedArray other ) {
                swap( other );
                
                return *this;
            }
            
            /**
             * Swaps the managed data and the ownership of it with @a other.
             *
             * @throw Nothing.
             */
            void swap( SharedArray& other ) {
                std::swap( data_, other.data_ );
                std::swap( referenceCount_, other.referenceCount_ );
            }
            
            
            /**
             * @attention Don't call if the array pointer is @c 0.
             */
            reference operator*() const {
                assert( 0 != data_ && "Unable to dereference a 0-pointer." );
                return *data_;
            }
            
            /*
             const_reference operator*() const {
                 return data_->data_.operator*();
             }
             */
            
            /**
             * @attention Don't call if the array pointer is @c 0.
             */
            pointer operator->() const {
                assert( 0 != data_ && "Unable to dereference a 0-pointer." );
                return data_;
            }
            
            
            /**
             * @attention Don't call if the array pointer is @c 0.
             */
            reference operator[]( size_type _index ) const {
                return data_[ _index ];
            }
            
            size_type useCount() const {
                return referenceCount_->referenceCount_;
            }
            
            /**
             * Get direct control of the raw managed pointer.
             *
             * @attention Don't delete the pointer behind the back of the smart
             *            pointer because this leads to undefined behavior.
             *
             * @attention Don't put the pointer under the management of this or
             *            another smart array pointer!
             */
            pointer get() const {
                return data_;
            }
            
            /**
             * Sets a new pointer to be managed by the smart array pointer.
             *
             * @attention Don't add a pointer that is already managed by another
             *            smart array pointer. Otherwise behavior is undefined.
             */
            void reset( T* _data = 0 ) {
                
                SharedArray( _data ).swap( *this );
            }
            
            
            /**
             * See http://Boost.org shared_ptr 
             * ( http://www.boost.org/libs/smart_ptr/shared_ptr.htm ).
             */
            typedef T* (SharedArray::*unspecified_bool_type)() const;
            
            /**
             * Automatic cast operator to enable use of a smart pointer inside a
             * conditional, for example to test if the smart pointer manages a @c 0
             * pointer.
             *
             * See http://Boost.org shared_ptr 
             * ( http://www.boost.org/libs/smart_ptr/shared_ptr.htm ).
                 */
            operator unspecified_bool_type () const {
                return 0 == data_ ? 0 : &SharedArray::get;
            }
            
            
            
            
            /**
             * Catches comparisons with @c 0. See Andrei Alexandrescu, 
             * Modern C++ Design, Addison-Wesley, 2002, p. 176.
             */
            // inline friend bool operator==( SharedPointer const& lhs, T const* rhs ) {
            //     return lhs.get() == rhs;
            // }
            
            
            /**
             * Catches comparisons with @c 0. See Andrei Alexandrescu, 
             * Modern C++ Design, Addison-Wesley, 2002, p. 176.
             */
            // inline friend bool operator==( T const* lhs, SharedPointer const& rhs ) {
            //     return lhs == rhs.get();
            // }
            
            /**
             * Catches comparisons with @c 0. See Andrei Alexandrescu, 
             * Modern C++ Design, Addison-Wesley, 2002, p. 176.
             */        
            // inline friend bool operator!=( SharedPointer const& lhs, T const* rhs ) {
            //     return !( lhs == rhs )
            // }
            
            /**
             * Catches comparisons with @c 0. See Andrei Alexandrescu, 
             * Modern C++ Design, Addison-Wesley, 2002, p. 176.
             */
            // inline friend bool operator!=( T const* lhs, SharedPointer const& rhs ) {
            //     return !( lhs == rhs );
            // }
            
            
            template< typename U >
                bool operator<( SharedArray< U > const& rhs ) {
                    // Because of sub-typing two different pointers might
                    // nonetheless point to the same class instance. Therefore use
                    // the reference count pointer for comparisons.
                    return referenceCount_ < rhs.referenceCount_;
                }
            
            
            /**
             * Calls @c static_cast for the pointer shared by @a sp while 
             * preserving the right use count.
             *
             * Naming scheme resembles that of the Boost library.
             */
            template< typename U >
                inline friend SharedArray< U > static_pointer_cast( SharedArray const& sp ) {
                    
                    return sp.static_pointer_cast< U >();
                }
            
            
            /**
             * Calls @c const_cast for the pointer shared by @a sp while 
             * preserving the right use count.
             *
             * Naming scheme resembles that of the Boost library.
             */
            template< typename U >
                inline friend SharedArray< U > const_pointer_cast( SharedArray const& sp ) {
                    
                    return sp.const_pointer_cast< U >();
                }
            
            /**
             * Calls @c dynamic_cast for the pointer shared by @a sp while 
             * preserving the right use count.
             *
             * Naming scheme resembles that of the Boost library.
             */
            template< typename U >
                inline friend SharedArray< U > dynamic_pointer_cast( SharedArray const& sp ) {                
                    return sp.dynamic_pointer_cast< U >();
                    
                }    
            
            
            
            
            
        private:
                
                /**
                 * Decreases the reference count of the managed pointer. If the 
                 * reference count reaches @c 0 the managed pointer is deleted, too.
                 * 
                 * @attention @c release and @c retain are used internally and only
                 *            public to allow developers to gain more control over
                 *            the smart pointer. But beware, if release is called and
                 *            the managed pointer is destroyed further calls to the 
                 *            dereference operators or even the destructor show
                 *            undefined behavior and might crash the application.
                 */
                void release() {
                    assert( 0 < referenceCount_->referenceCount_ && "Only call release for reference counts greater than 0." );
                    
                    --( referenceCount_->referenceCount_ );
                    if ( 0 == referenceCount_->referenceCount_ ) {
                        delete[] data_;
                        data_ = 0;
                        delete referenceCount_;
                        referenceCount_ = 0;
                    }
                }
            
            /**
             * Increases the reference count for the managed pointer.
             *
             * Mostly used internally, therefor beware of wrong usage that might
             * lead to memory leaks.
             */
            void retain() {
                ++( referenceCount_->referenceCount_ );
                
            }
            
            
            /**
             * Helper constructor used by the friend cast functions.
             */
            SharedArray( T* _data, SharedArrayReferenceCount* _referenceCount ) : data_( _data ), referenceCount_( _referenceCount ) {
                retain();
            }
            
            
            /**
             * Helper function for the friend function @c static_pointer_cast.
             *
             * It is needed because two @c SharedPointer classes with different
             * template parameters grand friendship to each other while a friend
             * function only is the friend of one of both classes and doesn't get 
             * access to the special constructor needed for casts of the other
             * class.
             */
            template< typename U >
                SharedArray< U > static_pointer_cast() const {
                    
                    return SharedArray< U >( static_cast< U* >( get() ), referenceCount_ );
                }
            
            /**
             * Helper function for the friend function @c const_pointer_cast.
             *
             * It is needed because two @c SharedPointer classes with different
             * template parameters grand friendship to each other while a friend
             * function only is the friend of one of both classes and doesn't get 
             * access to the special constructor needed for casts of the other
             * class.
             */
            template< typename U >
                SharedArray< U > const_pointer_cast() const {
                    
                    return SharedArray< U >( const_cast< U* >( get() ), referenceCount_ );
                }
            
            /**
             * Helper function for the friend function @c dynamic_pointer_cast.
             *
             * It is needed because two @c SharedPointer classes with different
             * template parameters grand friendship to each other while a friend
             * function only is the friend of one of both classes and doesn't get 
             * access to the special constructor needed for casts of the other
             * class.
             */
            template< typename U >
                inline SharedArray< U > dynamic_pointer_cast() const {
                    
                    U* castPointer = dynamic_cast< U* >( get() );
                    
                    if ( 0 == castPointer ) {
                        return SharedArray< U >();
                    }
                    
                    return SharedArray< U >( castPointer, referenceCount_ );
                } 
            
            
    private:
        pointer data_;
        SharedArrayReferenceCount* referenceCount_;
    }; // class SharedArray
    
    
    
    
    template< typename T, typename U >
        bool operator==( SharedArray< T > const& lhs, SharedArray< U > const& rhs ) {
            return lhs.get() == rhs.get();
        }
    
    // template< typename T, typename U >
    //     bool operator==( SharedArray< T > const& lhs, U const* rhs ) {
    //         return lhs.get == rhs;
    //     }
    
    // template< typename T, typename U >
    //     bool operator==( T const* lhs, SharedArray< U > const& rhs ) {
    //         return lhs == rhs.get();
    //     }
    
    
    template< typename T, typename U >
        bool operator!=( SharedArray< T > const& lhs, SharedArray< U > const& rhs ) {
            return !( lhs == rhs );
        }
    
    // template< typename T, typename U >
    //     bool operator!=( SharedArray< T > const& lhs, U const* rhs ) {
    //         return !( lhs == rhs );
    //     }
    
    // template< typename T, typename U >
    //     bool operator!=( T const* lhs, SharedArray< U > const& rhs ) {
    //         return !( lhs == rhs );
    //     }
    
    
    
    template< typename T >
        void swap( SharedArray< T >& lhs, SharedArray< T >& rhs ) {
            lhs.swap( rhs );
        }
    
    
} // namespace OpenSteer





#endif // OPENSTEER_SHAREDARRAY_H
