// Real-time annotations and the few audited primitives real-time code may call.
//
// ECHO_RT marks a function as [[clang::nonblocking]]. The build uses
// -Werror=function-effects, so such a function may not allocate, lock, throw, or
// call anything the compiler cannot prove is nonblocking too. The audio render
// path is marked with it. Write it after noexcept: `void f() noexcept ECHO_RT;`
#pragma once

#include <cmath>
#include <cstdint>

#include <mach/mach_time.h>

#define ECHO_RT [[clang::nonblocking]]

namespace rt {

// Hand-audited exceptions: the compiler cannot see inside these, so they are
// listed here and nowhere else.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfunction-effects"

// Reads the commpage clock; Core Audio itself calls it on the IO thread.
// Same tick domain as AudioTimeStamp::mHostTime and (in seconds) Metal's GPU times.
inline uint64_t nowTicks() noexcept ECHO_RT { return mach_absolute_time(); }

// Pure libm computation, no allocation or locking.
inline double sin(double x) noexcept ECHO_RT { return std::sin(x); }

#pragma clang diagnostic pop

}  // namespace rt
