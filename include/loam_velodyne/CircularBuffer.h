
// CircularBuffer.h

#ifndef LOAM_CIRCULAR_BUFFER_H
#define LOAM_CIRCULAR_BUFFER_H

#include <cstdlib>
#include <vector>

namespace loam {

/** \brief Simple circular buffer implementation for storing data history.
 *
 * @tparam T The buffer element type.
 */
template <class T>
class CircularBuffer
{
public:
    CircularBuffer(const std::size_t capacity = 200) :
        _capacity(capacity),
        _size(0),
        _startIdx(0),
        _buffer(new T[capacity]) { }

    ~CircularBuffer()
    {
        delete[] this->_buffer;
        this->_buffer = nullptr;
    }

    /** \brief Retrieve the buffer size.
     *
     * @return The buffer size
     */
    inline std::size_t size() const { return this->_size; }

    /** \brief Retrieve the buffer capacity.
     *
     * @return The buffer capacity
     */
    inline std::size_t capacity() const { return this->_capacity; }

    /** \brief Ensure that this buffer has at least the required capacity.
     *
     * @param reqCapacity The minimum required capacity
     */
    void ensureCapacity(const int reqCapacity)
    {
        if (reqCapacity <= 0 || this->_capacity >= reqCapacity)
            return;

        // Create new buffer and copy (valid) entries
        T* newBuffer = new T[reqCapacity];
        for (std::size_t i = 0; i < this->_size; ++i)
            newBuffer[i] = this->at(i);

        // Switch buffer pointers and delete old buffer
        T* oldBuffer = this->_buffer;
        this->_buffer = newBuffer;
        this->_startIdx = 0;

        delete[] oldBuffer;
    }

    /** \brief Check if the buffer is empty.
     *
     * @return true if the buffer is empty, false otherwise
     */
    inline bool empty() const { return this->_size == 0; }

    /** \brief Retrieve the i-th element of the buffer.
     *
     * @param i The buffer index
     * @return The element at the i-th position
     */
    inline const T& operator[](const std::size_t i) const
    { return this->_buffer[this->toBufferIndex(i)]; }

    inline const T& at(const std::size_t i) const
    { return this->_buffer[this->toBufferIndex(i)]; }

    /** \brief Retrieve the first (oldest) element of the buffer.
     *
     * @return The first element
     */
    inline const T& first() const
    { return this->_buffer[this->toBufferIndex(0)]; }

    /** \brief Retrieve the last (latest) element of the buffer.
     *
     * @return the last element
     */
    inline const T& last() const
    { return this->_buffer[this->toBufferIndex(this->_size - 1)]; }

    /** \brief Push a new element to the buffer.
     *
     * If the buffer reached its capacity, the oldest element is overwritten.
     *
     * @param element The element to push
     */
    void push(const T& element)
    {
        if (this->_size < this->_capacity) {
            this->_buffer[this->_size] = element;
            this->_size++;
        } else {
            this->_buffer[this->_startIdx] = element;
            this->_startIdx = (this->_startIdx + 1) % this->_capacity;
        }
    }

private:
    inline std::size_t toBufferIndex(const std::size_t idx) const
    { return (this->_startIdx + idx) % this->_capacity; }

private:
    // Buffer capacity
    std::size_t _capacity;
    // Current buffer size
    std::size_t _size;
    // Current start index
    std::size_t _startIdx;
    // Internal element buffer
    T*          _buffer;
};

} // namespace loam

#endif //LOAM_CIRCULAR_BUFFER_H
