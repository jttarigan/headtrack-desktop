// Shared-memory frame ring from HeadTrackRecorder (writer) to the MediaPipe
// tracker, headtrack/analysis/mp_live.py (reader). The Python side reads the
// same layout with struct formats "<6IQ" (header) and "<4Q" (slot header).
// After each frame the writer sends one byte down <session>/frames.fifo.
#pragma once

#include <stdint.h>

#define HT_FRAMES_SHM "/eht_frames"
#define HT_FRAMES_MAGIC 0x31524645u  // "EFR1"
#define HT_FRAME_SLOTS 4u

struct HtFramesHeader {  // 32 bytes at offset 0
    uint32_t magic;
    uint32_t slots;
    uint32_t slotBytes;  // slot header + stride * height
    uint32_t width, height;
    uint32_t stride;     // bytes per row, BGRA
    uint64_t latestSeq;  // newest complete frame
};

struct HtFrameSlot {  // 32 bytes, then stride * height bytes of BGRA pixels
    uint64_t seq;     // 0 while being written; frame n lives in slot n % slots
    uint64_t ptsNs;        // capture time, Mac host clock
    uint64_t deliveredNs;  // when the camera handed it to the recorder
    uint64_t sharedNs;     // when the copy into this slot finished
};
