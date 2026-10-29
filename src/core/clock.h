// Host clock. One time base for CPU, audio and GPU timestamps:
// mach_absolute_time ticks, which in seconds equal CACurrentMediaTime() and
// MTLCommandBuffer.GPUStartTime / GPUEndTime.
#pragma once

#include <cstdint>
#include <string>

#include "core/rt.h"

namespace core::clock {

inline uint64_t now() noexcept ECHO_RT { return rt::nowTicks(); }

double toSeconds(uint64_t ticks);
inline double toMs(uint64_t ticks) { return toSeconds(ticks) * 1e3; }
// Signed difference to - from, in milliseconds.
double deltaMs(uint64_t from, uint64_t to);
uint64_t fromSeconds(double seconds);

// Wall-clock UTC, e.g. "2026-10-12T04:31:05.123Z".
std::string utcIsoNow();
// Wall-clock UTC for directory names, e.g. "20261012T043105Z".
std::string utcCompactNow();

}  // namespace core::clock
