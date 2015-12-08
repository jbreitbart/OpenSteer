#ifndef OPENSTEER_MATRIX_H
#define OPENSTEER_MATRIX_H


// Include std::fill, std::copy, std::swap_ranges
#include <algorithm>

// Include assert
#include <cassert>

// Include std::distance
#include <iterator>

// Include std::memcpy
// @todo Needed?
// #include <cstring>

// Include OpenSteer::size_t, OpenSteer::diff_t
#include "OpenSteer/StandardTypes.h"

// Include OpenSteer::Vec3
#include "OpenSteer/Vec3.h"

#include "cupp/device.h"
#include "OpenSteer/deviceT/Matrix.h"

namespace OpenSteer {
    
    /**
     * 4x4 matrix storing its elements in column-major order like OpenGL
     * matrices to allow easy data exchange with OpenGL.
     *
     * Provides an interface like many of the C++ stl containers.
     */
    class Matrix {
    public:
        typedef deviceT::Matrix device_type;
        typedef Matrix          host_type;

        device_type transform(const cupp::device &/*d*/) {
        	device_type returnee = {{elements_[0], elements_[1], elements_[2], elements_[3],
        	                         elements_[4], elements_[5], elements_[6], elements_[7],
        	                         elements_[8], elements_[9], elements_[10], elements_[11],
        	                         elements_[12], elements_[13], elements_[14], elements_[15]
        	                       }};
        	return returnee;
        }

        Matrix& operator= (const device_type &rhs) {
            for (int i=0; i<15; ++i) {
                  elements_[i] = rhs.elements_[i];
            }
            return *this;
        }
        
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef float value_type;
        typedef value_type& reference;
        typedef value_type const& const_reference;
        typedef value_type* pointer;
        typedef value_type const* const_pointer;
        typedef pointer iterator;
        typedef const_pointer const_iterator;
        
        /**
         * Constructs a matrix with uninitialized elements. Elements can contain
         * arbitrary values.
         */
        Matrix() {
            // Nothing to do.
        }
        
        /**
         * Constructs a matrix and sets every element to @a initValue.
         */
        explicit Matrix( value_type const& initValue ) {
            assign( initValue );
        }
        
        
        Matrix( Matrix const& other ) {
            // @todo What is faster?
            std::copy( other.begin(), other.end(), begin() );
            // std::memcpy( elements_, other.elements_, size() * sizeof( value_type ) );
        }
        
        
        /**
         * Constructs the matrix and assigns @a e00 to the element in row @c 0
         * and column @c 0, assigns @a e01 to the element in row @c 0 and 
         * column @c 1, and so on.
         */
        Matrix( value_type e00, value_type e01, value_type e02, value_type e03,
                value_type e10, value_type e11, value_type e12, value_type e13,
                value_type e20, value_type e21, value_type e22, value_type e23,
                value_type e30, value_type e31, value_type e32, value_type e33 ) {
                
            assign( e00, e01, e02, e03,
                    e10, e11, e12, e13,
                    e20, e21, e22, e23,
                    e30, e31, e32, e33 );
            
        }
        
        /**
         * Constructs a matrix using the given vectors for the columns.
         *
         * @todo Check the implementation - it is for sure wrong.
         */
        Matrix( Vec3 const& column0, Vec3 const& column1, Vec3 const& column2, Vec3 const& column3 ) {
            
            assign( column0, column1, column2, column3 );
        }
        
        /**
         * Constructs a matrix and copies @a first to @a last (excluding @a last)
         * into the matrix elements in column-major order (first the first
         * column is filled, then the second one, and so on).
         *
         * @attention <code>last - first </code> must be lesser or equal to 
         *            @c size().
         */
        template< typename InputIterator >
            Matrix( InputIterator first, InputIterator last ) {
                assign( first, last );
            }
        
        
        Matrix& operator=( Matrix other ) {
            swap( other );
            return *this;
        }
        
        /**
         * Swaps the content with @a other.
         */
        void swap( Matrix& other ) {
            std::swap_ranges( other.begin(), other.end(), begin() );
        }
        
        
        /**
         * Copies the elements in the range @a first to @a last (excluding @a last)
         * in column-major order into the matrix. Elements that aren't replaced
         * by copying the range keep their old values.
         */
        template< typename InputIterator >
            void assign( InputIterator first, InputIterator last ) {
                assert( std::distance( first, last ) <= size() && "Too many elements in the range first to last." );
                std::copy( first, last, begin() );
            }
        
        /**
         * Assigns @a initValue to all elements.
         */
        void assign( value_type const& initValue ) {
            std::fill( begin(), end(), initValue );
        }
        
        /**
         * Assigns @a e00 to the element in row @c 0
         * and column @c 0, assigns @a e01 to the element in row @c 0 and 
         * column @c 1, and so on.
         */
        void assign( value_type e00, value_type e01, value_type e02, value_type e03,
                     value_type e10, value_type e11, value_type e12, value_type e13,
                     value_type e20, value_type e21, value_type e22, value_type e23,
                     value_type e30, value_type e31, value_type e32, value_type e33 ) {
            
            elements_[  0 ] = e00;
            elements_[  1 ] = e10;
            elements_[  2 ] = e20;
            elements_[  3 ] = e30;
            
            elements_[  4 ] = e01;
            elements_[  5 ] = e11;
            elements_[  6 ] = e21;
            elements_[  7 ] = e31;
            
            elements_[  8 ] = e02;
            elements_[  9 ] = e12;
            elements_[ 10 ] = e22;
            elements_[ 11 ] = e32;
            
            elements_[ 12 ] = e03;
            elements_[ 13 ] = e13;
            elements_[ 14 ] = e23;
            elements_[ 15 ] = e33;
            
        }
        
        
        /**
         * Assigns the vectors to the columns.
         *
         * @todo Check if this works correctly - there is much doubt here!
         */
        void assign( Vec3 const& column0, Vec3 const& column1, Vec3 const& column2, Vec3 const& column3 ) {
            elements_[  0 ] = column0.x;
            elements_[  1 ] = column0.y;
            elements_[  2 ] = column0.z;
            elements_[  3 ] = value_type( 0 );
            
            elements_[  4 ] = column1.x;
            elements_[  5 ] = column1.y;
            elements_[  6 ] = column1.z;
            elements_[  7 ] = value_type( 0 );
            
            elements_[  8 ] = column2.x;
            elements_[  9 ] = column2.y;
            elements_[ 10 ] = column2.z;
            elements_[ 11 ] = value_type( 0 );
            
            elements_[ 12 ] = column3.x;
            elements_[ 13 ] = column3.y;
            elements_[ 14 ] = column3.z;
            elements_[ 15 ] = value_type( 1 );
        }
        
        
        void assignColumn( size_type _columnIndex, float _c0, float _c1, float _c2, float _c3 ) {
            assert( _columnIndex < columnCount() && "_columnIndex out of range." );
            
            size_type const columnIndex = _columnIndex * 4;

            elements_[ columnIndex ]    = _c0;
            elements_[ columnIndex + 1] = _c1;
            elements_[ columnIndex + 2] = _c2;
            elements_[ columnIndex + 3] = _c3;
        }
        
        
        
        
        iterator begin() {
            return elements_;
        }
        
        const_iterator begin() const {
            return elements_;
        }
        
        iterator end() {
            return &elements_[ size() ];
        }
        
        const_iterator end() const{
            return &elements_[ size() ];
        }
        
        /**
         * Returns the number of elements of the matrix.
         */
        size_type size() const {
            return 16;
        }
        
        /**
         * Returns the number of elements of the matrix.
         */
        size_type elementCount() const {
            return size();
        }
        
        /**
         * Returns the maximum size of a matrix which is also its @a size().
         */
        size_type max_size() const {
            return size();
        }
        
        /**
         * Always returns @c false because a matrix has a fixed set of elements.
         */
        bool empty() const {
            return false;
        }
        
        /**
         * Returns the number of columns of the matrix.
         */
        size_type columnCount() const {
            return 4;
        }
        
        /**
         * Returns the number of rows of the matrix.
         */
        size_type rowCount() const {
            return columnCount();
        }
        
        /**
         * Returns element number @a index. The elements are stored in column-
         * major order therefore an index of @c 1 references the first element
         * in row @c 0. Index @c 4 references the second element in row @c 0,
         * and so on.
         */
        reference operator[]( size_type index ) {
            assert( index < size() && "index out of range." );
            return elements_[ index ];
        }
        
        /**
         * Returns element number @a index. The elements are stored in column-
         * major order therefore an index of @c 1 references the first element
         * in row @c 0. Index @c 4 references the second element in row @c 0,
         * and so on.
         */
        const_reference operator[]( size_type index ) const{
            assert( index < size() && "index out of range." );
            return elements_[ index ];
        }
        
        /**
         * Returns the element in row @a rowIndex and column @a columnIndex.
         */
        reference operator()( size_type rowIndex, size_type columnIndex ) {
            return operator[]( rowIndex + ( rowCount() * columnIndex ) );
        }
        
        /**
         * Returns the element in row @a rowIndex and column @a columnIndex.
         */
        const_reference operator()( size_type rowIndex, size_type columnIndex ) const{
            return operator[]( rowIndex + ( rowCount() * columnIndex ) );
        }        
        
        
        reference front() {
            return operator[]( 0 );
        }
        
        
        const_reference front() const{
            return operator[]( 0 );
        }
        
        reference back() {
            return operator[]( size() - 1 );
        }
        
        
        const_reference back() const {
            return operator[]( size() - 1 );
        }
        
        /**
         * Returns a constant pointer to the element data.
         *
         * Usefull to pass the matrix data to libraries like @b OpenGL.
         */
        const_pointer data() const {
            return elements_;
        }
        
    private:
        value_type elements_[ 16 ];
        
    }; // class Matrix
    
    
    /**
     * Swaps the content of @a lsh and @a rhs.
     */
    inline void swap( Matrix& lhs, Matrix& rhs ) {
        lhs.swap( rhs );
    }
    
    
    extern Matrix const identityMatrix;
    
    
    // @todo Write unit test. Also provide a batch function to add matrices.
    Matrix& operator+=( Matrix& _lhs, Matrix const& _rhs );

    // @todo Write unit test. Also provide a batch function to add matrices.    
    Matrix operator+( Matrix const& _lhs, Matrix const& _rhs );
    


    // @todo Write a unit test. Also provide a batch function to multiply matrices.
    Matrix operator*( Matrix const& _lhs, Matrix const& _rhs );
    
    
    
    
} // namespace OpenSteer


#endif // OPENSTEER_MATRIX_H
