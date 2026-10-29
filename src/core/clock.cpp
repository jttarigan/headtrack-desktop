#include "core/clock.h"

#include <chrono>
#include <cstdio>
#include <ctime>

namespace core::clock {
namespace {

double secondsPerTick() {
    static const double value = [] {
        mach_timebase_info_data_t tb{};
        mach_timebase_info(&tb);
        return double(tb.numer) / double(tb.denom) * 1e-9;
    }();
    return value;
}

std::string formatUtc(const char* format, bool withMillis) {
    using namespace std::chrono;
    const auto t = system_clock::now();
    const std::time_t secs = system_clock::to_time_t(t);
    std::tm tm{};
    gmtime_r(&secs, &tm);
    char buf[64];
    size_t n = std::strftime(buf, sizeof buf, format, &tm);
    std::string out(buf, n);
    if (withMillis) {
        const auto ms = duration_cast<milliseconds>(t.time_since_epoch()).count() % 1000;
        char frac[8];
        std::snprintf(frac, sizeof frac, ".%03dZ", int(ms));
        out += frac;
    }
    return out;
}

}  // namespace

double toSeconds(uint64_t ticks) { return double(ticks) * secondsPerTick(); }

double deltaMs(uint64_t from, uint64_t to) {
    return to >= from ? toMs(to - from) : -toMs(from - to);
}

uint64_t fromSeconds(double seconds) {
    return seconds <= 0 ? 0 : uint64_t(seconds / secondsPerTick());
}

std::string utcIsoNow() { return formatUtc("%Y-%m-%dT%H:%M:%S", true); }
std::string utcCompactNow() { return formatUtc("%Y%m%dT%H%M%SZ", false); }

}  // namespace core::clock
