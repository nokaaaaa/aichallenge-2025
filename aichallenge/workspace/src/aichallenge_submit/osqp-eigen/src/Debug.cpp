/**
 * @file Debug.cpp
 * @copyright Released under the terms of the BSD 3-Clause License
 * @date 2023
 */
#include <OsqpEigen/Debug.hpp>
#include <streambuf>

namespace OsqpEigen
{

// NullBuffer that discards all output
class NullBuffer : public std::streambuf
{
protected:
    int overflow(int c) override { return c; }
};

// NullStream that uses NullBuffer
class NullStream : public std::ostream
{
public:
    NullStream()
        : std::ostream(&m_sb) {}

    // コピーコンストラクタと代入演算子を削除（使われないため）
    NullStream(const NullStream&) = delete;
    NullStream& operator=(const NullStream&) = delete;

private:
    NullBuffer m_sb;
};

// 明示的な NullStream オブジェクト
NullStream theStream;

std::ostream& debugStream()
{
#ifdef OSQP_EIGEN_DEBUG_OUTPUT
    return std::cerr;
#else
    return theStream;
#endif
}

} // namespace OsqpEigen
