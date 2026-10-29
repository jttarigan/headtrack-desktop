// UDP messages between HeadTrackRecorder (Mac) and HeadTrackRef (iPhone).
// Both ends are arm64 Apple devices, so packed little-endian structs are sent
// as they are. Times are each device's own mach_absolute_time in nanoseconds;
// the analysis maps iPhone times onto the Mac clock from the ping exchange.
#pragma once

#include <stdint.h>

#define HT_MAGIC 0x31544845u  // "EHT1"
#define HT_BONJOUR_TYPE "_ehtref._udp"

enum HtType : uint32_t {
    HT_PING = 1,  // Mac -> iPhone, 10 Hz. Also tells the iPhone where to stream.
    HT_PONG = 2,  // iPhone -> Mac, answer to a ping
    HT_POSE = 3,  // iPhone -> Mac, one per ARKit frame
};

#pragma pack(push, 1)

struct HtHeader {
    uint32_t magic;
    uint32_t type;
};

struct HtPing {
    HtHeader h;
    uint32_t seq;
    uint64_t t1;  // Mac, at send
};

struct HtPong {
    HtHeader h;
    uint32_t seq;
    uint64_t t1;  // echoed
    uint64_t t2;  // iPhone, at receive
    uint64_t t3;  // iPhone, at send
};

struct HtPose {
    HtHeader h;
    uint32_t seq;          // frames since app start
    uint64_t captureNs;    // ARFrame.timestamp (camera capture time), iPhone clock
    uint64_t deliveredNs;  // iPhone clock when ARKit handed the frame to the app
    uint64_t sentNs;       // iPhone clock just before sending
    uint8_t tracked;       // a face anchor is present and isTracked
    uint8_t pad[3];
    float fps;             // ARKit video format frame rate
    // Head pose in the camera's frame: camera.transform^-1 * faceAnchor.transform.
    // Quaternion (x, y, z, w) and position in metres. ARKit camera axes.
    double qx, qy, qz, qw;
    double px, py, pz;
};

#pragma pack(pop)
