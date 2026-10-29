// HeadTrackRecorder: the side paper's recording app on the Mac (HEADTRACK_STUDY.md).
// One session = one folder of raw logs, all times on the Mac's host clock
// (mach_absolute_time in ns) except the iPhone's own times, which the analysis
// maps across using sync.csv.
//
//   airpods.csv  AirPods head motion (CMHeadphoneMotionManager)
//   camera.csv   Mac camera frames, with Apple Vision face roll/yaw/pitch
//   ref.csv      iPhone ARKit head poses (HeadTrackRef), with Mac receive times
//   sync.csv     ping exchange with the iPhone (t1..t4)
//   events.csv   protocol prompts, connections, drops, thermal changes
//   session.json settings, devices, versions, end summary (and the laser targets)
//   log.txt      human-readable log
//
// Camera frames are also published to shared memory (/eht_frames) with a FIFO
// wake-up, so the MediaPipe tracker (headtrack/analysis/mp_live.py) sees exactly
// the same frames without opening the camera itself.
//
// With --targets (laser check, headtrack/protocols/laser_check.md) the marks are
// drawn full screen on the display they were computed for, and the current
// step's mark is highlighted.
//
// Launched by headtrack/run_session.sh through `open`, so the recorder owns its
// camera, motion and local-network permissions.

#import <AVFoundation/AVFoundation.h>
#import <AppKit/AppKit.h>
#import <CoreMotion/CoreMotion.h>
#import <Network/Network.h>
#import <Vision/Vision.h>
#include <CoreAudio/CoreAudio.h>
#include <fcntl.h>
#include <mach/mach_time.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common/frames.h"
#include "common/wire.h"
#include "core/clock.h"
#include "core/json.h"

using core::json::Value;

namespace {

// ---- settings -------------------------------------------------------------

struct Settings {
    std::string outDir;
    std::string protocolPath;  // empty = free run until Esc or --duration
    std::string targetsPath;   // laser check marks (laser_targets.py); empty = none
    std::string renderPath;    // --render-targets: draw the target screen to a PNG and exit
    double durationS = 0;      // 0 = no limit
    bool airpods = true, camera = true, vision = true, shareFrames = true, reference = true, speak = false;
    int width = 1280, height = 720, fps = 30;
    double photosS = 0;  // --photos: save the window as PNG every N s (test-process figures); 0 = off
};

bool onOff(const char* v) { return std::strcmp(v, "on") == 0 || std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0; }

Settings gSet;

// ---- time and logging ------------------------------------------------------

mach_timebase_info_data_t gTimebase;
uint64_t ticksToNs(uint64_t t) { return t * gTimebase.numer / gTimebase.denom; }
uint64_t nowNs() { return ticksToNs(mach_absolute_time()); }
uint64_t gStartNs = 0;

dispatch_queue_t gLogQueue;  // every file write goes through this serial queue
bool gFilesClosed = false;   // touched on gLogQueue only

struct Csv {
    FILE* f = nullptr;
    void open(const char* name, const char* header) {
        f = std::fopen((gSet.outDir + "/" + name).c_str(), "w");
        std::fputs(header, f);
        std::fputc('\n', f);
    }
    void row(std::string line) {
        FILE* file = f;
        dispatch_async(gLogQueue, ^{
            if (gFilesClosed) return;
            std::fputs(line.c_str(), file);
            std::fputc('\n', file);
        });
    }
};

Csv gAirpodsCsv, gCameraCsv, gRefCsv, gSyncCsv, gEventsCsv;
FILE* gLogFile = nullptr;

std::string fmt(const char* f, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

void say(const std::string& msg) {
    const std::string line = core::clock::utcIsoNow() + "  " + msg + "\n";
    std::fputs(line.c_str(), stdout);
    std::fflush(stdout);
    dispatch_async(gLogQueue, ^{
        if (!gFilesClosed && gLogFile) std::fputs(line.c_str(), gLogFile);
    });
}

void event(const std::string& kind, const std::string& detail) {
    const uint64_t t = nowNs();
    std::string d = detail;
    for (char& c : d)
        if (c == ',' || c == '\n') c = ';';
    gEventsCsv.row(fmt("%llu,%s,%s", (unsigned long long)t, kind.c_str(), d.c_str()));
    say(kind + ": " + detail);
}

// ---- status counters (read by the UI timer) -------------------------------

std::atomic<uint64_t> gAirpodsSamples{0}, gCameraFrames{0}, gCameraDrops{0}, gVisionFaces{0}, gRefPoses{0},
    gRefTracked{0}, gPongs{0}, gFramesShared{0}, gLastRttNs{0};
std::atomic<bool> gReaderAttached{false}, gRefConnected{false};

// ---- AirPods ---------------------------------------------------------------

CMHeadphoneMotionManager* gMotion;
NSOperationQueue* gMotionQueue;

}  // namespace

@interface MotionDelegate : NSObject <CMHeadphoneMotionManagerDelegate>
@end
@implementation MotionDelegate
- (void)headphoneMotionManagerDidConnect:(CMHeadphoneMotionManager*)m {
    event("airpods", "connected");
}
- (void)headphoneMotionManagerDidDisconnect:(CMHeadphoneMotionManager*)m {
    event("airpods", "disconnected");
}
@end

namespace {

MotionDelegate* gMotionDelegate;

void startAirpods() {
    gAirpodsCsv.open("airpods.csv",
                     "arrival_ns,timestamp_ns,qx,qy,qz,qw,rot_x,rot_y,rot_z,grav_x,grav_y,grav_z,acc_x,acc_y,acc_z,"
                     "sensor_location");
    gMotion = [CMHeadphoneMotionManager new];
    gMotionDelegate = [MotionDelegate new];
    gMotion.delegate = gMotionDelegate;
    [gMotion startConnectionStatusUpdates];
    gMotionQueue = [NSOperationQueue new];
    gMotionQueue.maxConcurrentOperationCount = 1;
    gMotionQueue.qualityOfService = NSQualityOfServiceUserInteractive;
    say(fmt("airpods: authorization %ld, device motion available %d", long(CMHeadphoneMotionManager.authorizationStatus),
            int(gMotion.deviceMotionAvailable)));
    [gMotion startDeviceMotionUpdatesToQueue:gMotionQueue
                                 withHandler:^(CMDeviceMotion* m, NSError* error) {
                                     const uint64_t arrival = nowNs();
                                     if (error) {
                                         event("airpods_error", fmt("%s %ld: %s", error.domain.UTF8String,
                                                                    long(error.code),
                                                                    error.localizedDescription.UTF8String));
                                         return;
                                     }
                                     if (!m) return;
                                     const CMQuaternion q = m.attitude.quaternion;
                                     long loc = -1;
                                     if ([m respondsToSelector:@selector(sensorLocation)])
                                         loc = [[m valueForKey:@"sensorLocation"] longValue];
                                     gAirpodsCsv.row(fmt(
                                         "%llu,%llu,%.9f,%.9f,%.9f,%.9f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%ld",
                                         (unsigned long long)arrival,
                                         (unsigned long long)llround(m.timestamp * 1e9), q.x, q.y, q.z, q.w,
                                         m.rotationRate.x, m.rotationRate.y, m.rotationRate.z, m.gravity.x,
                                         m.gravity.y, m.gravity.z, m.userAcceleration.x, m.userAcceleration.y,
                                         m.userAcceleration.z, loc));
                                     gAirpodsSamples++;
                                 }];
}

// ---- shared frames for MediaPipe ------------------------------------------

struct FrameShare {
    bool tried = false;
    int fd = -1;
    uint8_t* base = nullptr;
    size_t bytes = 0;
    int fifo = -1;
    std::string fifoPath;
    uint64_t seq = 0;
} gShare;

void shareInit(uint32_t width, uint32_t height, uint32_t stride) {
    shm_unlink(HT_FRAMES_SHM);
    gShare.fd = shm_open(HT_FRAMES_SHM, O_CREAT | O_RDWR, 0600);
    const size_t slotBytes = sizeof(HtFrameSlot) + size_t(stride) * height;
    gShare.bytes = sizeof(HtFramesHeader) + HT_FRAME_SLOTS * slotBytes;
    if (gShare.fd < 0 || ftruncate(gShare.fd, off_t(gShare.bytes)) != 0) {
        say("frame share: shm_open/ftruncate failed; MediaPipe will get no frames");
        return;
    }
    gShare.base = (uint8_t*)mmap(nullptr, gShare.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, gShare.fd, 0);
    auto* h = (HtFramesHeader*)gShare.base;
    *h = HtFramesHeader{HT_FRAMES_MAGIC, HT_FRAME_SLOTS, uint32_t(slotBytes), width, height, stride, 0};
    gShare.fifoPath = gSet.outDir + "/frames.fifo";
    mkfifo(gShare.fifoPath.c_str(), 0600);
    say(fmt("frame share: %s, %u slots of %zu bytes, wake-up fifo %s", HT_FRAMES_SHM, HT_FRAME_SLOTS, slotBytes,
            gShare.fifoPath.c_str()));
}

// Called on the camera queue. Copies the frame into the next slot, then wakes
// the reader. A reader that is not there yet costs one failed open() per frame.
void shareFrame(CVPixelBufferRef pb, uint64_t ptsNs, uint64_t deliveredNs) {
    const size_t stride = CVPixelBufferGetBytesPerRow(pb), height = CVPixelBufferGetHeight(pb);
    if (!gShare.tried) {
        gShare.tried = true;
        shareInit(uint32_t(CVPixelBufferGetWidth(pb)), uint32_t(height), uint32_t(stride));
    }
    if (!gShare.base) return;
    auto* h = (HtFramesHeader*)gShare.base;
    if (stride != h->stride || height != h->height) return;
    const uint64_t seq = ++gShare.seq;
    auto* slot = (HtFrameSlot*)(gShare.base + sizeof(HtFramesHeader) + (seq % HT_FRAME_SLOTS) * h->slotBytes);
    __atomic_store_n(&slot->seq, 0, __ATOMIC_RELEASE);  // mark as being written
    CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    std::memcpy(slot + 1, CVPixelBufferGetBaseAddress(pb), stride * height);
    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    slot->ptsNs = ptsNs;
    slot->deliveredNs = deliveredNs;
    slot->sharedNs = nowNs();
    __atomic_store_n(&slot->seq, seq, __ATOMIC_RELEASE);
    __atomic_store_n(&h->latestSeq, seq, __ATOMIC_RELEASE);
    gFramesShared++;
    if (gShare.fifo < 0) gShare.fifo = open(gShare.fifoPath.c_str(), O_WRONLY | O_NONBLOCK);
    if (gShare.fifo >= 0) {
        const char b = 1;
        if (write(gShare.fifo, &b, 1) < 0 && errno == EPIPE) {
            close(gShare.fifo);
            gShare.fifo = -1;
            gReaderAttached = false;
            event("mediapipe", "reader detached");
        } else if (!gReaderAttached.exchange(true)) {
            event("mediapipe", "reader attached");
        }
    }
}

void shareClose() {
    if (gShare.fifo >= 0) close(gShare.fifo);  // reader sees EOF and exits
    if (gShare.base) munmap(gShare.base, gShare.bytes);
    if (gShare.fd >= 0) close(gShare.fd);
    shm_unlink(HT_FRAMES_SHM);
    if (!gShare.fifoPath.empty()) unlink(gShare.fifoPath.c_str());
}

}  // namespace

// ---- camera and Apple Vision ----------------------------------------------

@interface CameraSink : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property(nonatomic, strong) AVCaptureSession* session;
@end

@implementation CameraSink {
    VNDetectFaceRectanglesRequest* _request;
}

- (instancetype)init {
    self = [super init];
    _request = [VNDetectFaceRectanglesRequest new];
    _request.revision = VNDetectFaceRectanglesRequestRevision3;
    return self;
}

- (void)captureOutput:(AVCaptureOutput*)output
    didOutputSampleBuffer:(CMSampleBufferRef)sb
           fromConnection:(AVCaptureConnection*)connection {
    const uint64_t delivered = nowNs();
    CMClockRef host = CMClockGetHostTimeClock();
    CMClockRef sync = self.session.synchronizationClock ?: host;
    const CMTime hostPts = CMSyncConvertTime(CMSampleBufferGetPresentationTimeStamp(sb), sync, host);
    const uint64_t pts = ticksToNs(CMClockConvertHostTimeToSystemUnits(hostPts));
    CVPixelBufferRef pb = CMSampleBufferGetImageBuffer(sb);
    const uint64_t seq = gCameraFrames++;
    if (seq == 0)
        event("camera", fmt("first frame %zux%zu, %zu bytes per row", CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb),
                            CVPixelBufferGetBytesPerRow(pb)));
    if (gSet.shareFrames) shareFrame(pb, pts, delivered);

    std::string vision = ",,0,,,,,,,,";
    if (gSet.vision) {
        const uint64_t v0 = nowNs();
        VNImageRequestHandler* handler = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pb options:@{}];
        NSError* err = nil;
        [handler performRequests:@[ _request ] error:&err];
        const uint64_t v1 = nowNs();
        VNFaceObservation* best = nil;
        for (VNFaceObservation* f in _request.results)
            if (!best || f.boundingBox.size.width > best.boundingBox.size.width) best = f;
        if (best) {
            gVisionFaces++;
            const CGRect b = best.boundingBox;
            vision = fmt("%llu,%llu,1,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f,%.5f,%.4f", (unsigned long long)v0,
                         (unsigned long long)v1, best.yaw ? best.yaw.doubleValue * 180 / M_PI : NAN,
                         best.pitch ? best.pitch.doubleValue * 180 / M_PI : NAN,
                         best.roll ? best.roll.doubleValue * 180 / M_PI : NAN, b.origin.x, b.origin.y, b.size.width,
                         b.size.height, best.confidence);
        } else {
            vision = fmt("%llu,%llu,0,,,,,,,,", (unsigned long long)v0, (unsigned long long)v1);
        }
    }
    gCameraCsv.row(fmt("%llu,%llu,%llu,%s", (unsigned long long)seq, (unsigned long long)pts,
                       (unsigned long long)delivered, vision.c_str()));
}

- (void)captureOutput:(AVCaptureOutput*)output
    didDropSampleBuffer:(CMSampleBufferRef)sb
         fromConnection:(AVCaptureConnection*)connection {
    gCameraDrops++;
}

@end

namespace {

CameraSink* gCameraSink;
dispatch_queue_t gCameraQueue;

std::string fourCC(FourCharCode v) {
    char c[5] = {char(v >> 24), char(v >> 16), char(v >> 8), char(v), 0};
    return c;
}

// Opens the built-in camera at the requested size and frame rate, Center Stage
// off. Returns a description for session.json, or an "error" key.
Value startCamera() {
    auto info = Value::object();
    AVCaptureDevice* dev = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    if (!dev) {
        info.set("error", Value::string("no camera"));
        return info;
    }
    info.set("device", Value::string(dev.localizedName.UTF8String));
    info.set("model", Value::string(dev.modelID.UTF8String));
    if (@available(macOS 12.3, *)) {
        AVCaptureDevice.centerStageControlMode = AVCaptureCenterStageControlModeApp;
        AVCaptureDevice.centerStageEnabled = NO;
        info.set("center_stage_enabled", Value::boolean(AVCaptureDevice.centerStageEnabled));
    }
    if (@available(macOS 12.0, *)) {
        info.set("portrait_effect_active", Value::boolean(dev.portraitEffectActive));
    }
    if (@available(macOS 13.0, *)) {
        info.set("studio_light_active", Value::boolean(dev.studioLightActive));
    }
    // Pick the format matching the size with the highest frame rate.
    AVCaptureDeviceFormat* chosen = nil;
    double chosenMax = 0;
    auto formats = Value::array();
    for (AVCaptureDeviceFormat* f in dev.formats) {
        const CMVideoDimensions d = CMVideoFormatDescriptionGetDimensions(f.formatDescription);
        double maxFps = 0;
        for (AVFrameRateRange* r in f.videoSupportedFrameRateRanges) maxFps = std::max(maxFps, r.maxFrameRate);
        formats.push(Value::string(fmt("%dx%d %s max %.0f fps", d.width, d.height,
                                       fourCC(CMFormatDescriptionGetMediaSubType(f.formatDescription)).c_str(), maxFps)));
        if (d.width == gSet.width && d.height == gSet.height && maxFps + 0.01 >= gSet.fps && maxFps > chosenMax) {
            chosen = f;
            chosenMax = maxFps;
        }
    }
    info.set("formats", std::move(formats));
    if (!chosen) {
        info.set("error", Value::string(fmt("no %dx%d format at %d fps", gSet.width, gSet.height, gSet.fps)));
        return info;
    }
    AVCaptureSession* session = [AVCaptureSession new];
    NSError* err = nil;
    AVCaptureDeviceInput* input = [AVCaptureDeviceInput deviceInputWithDevice:dev error:&err];
    if (!input || ![session canAddInput:input]) {
        info.set("error", Value::string(err ? err.localizedDescription.UTF8String : "cannot add camera input"));
        return info;
    }
    [session beginConfiguration];
    [session addInput:input];
    // The session's preset decides the format unless one matches; a size
    // preset is the reliable way on macOS (setting activeFormat did not stick).
    AVCaptureSessionPreset preset = gSet.width == 640 && gSet.height == 480     ? AVCaptureSessionPreset640x480
                                    : gSet.width == 1280 && gSet.height == 720  ? AVCaptureSessionPreset1280x720
                                    : gSet.width == 1920 && gSet.height == 1080 ? AVCaptureSessionPreset1920x1080
                                                                                : nil;
    if (preset && [session canSetSessionPreset:preset]) session.sessionPreset = preset;
    AVCaptureVideoDataOutput* out = [AVCaptureVideoDataOutput new];
    out.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)};
    out.alwaysDiscardsLateVideoFrames = YES;
    gCameraSink = [CameraSink new];
    gCameraSink.session = session;
    gCameraQueue = dispatch_queue_create(
        "camera", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
    [out setSampleBufferDelegate:gCameraSink queue:gCameraQueue];
    [session addOutput:out];
    [session commitConfiguration];
    // Only after the input is committed: adding an input lets the session's
    // preset choose the format (it picked 1920x1080 when this was set earlier).
    if ([dev lockForConfiguration:&err]) {
        if (!preset || session.sessionPreset != preset) dev.activeFormat = chosen;
        dev.activeVideoMinFrameDuration = CMTimeMake(1, gSet.fps);
        dev.activeVideoMaxFrameDuration = CMTimeMake(1, gSet.fps);
        [dev unlockForConfiguration];
    } else {
        say("camera: lockForConfiguration failed: " + std::string(err.localizedDescription.UTF8String));
    }
    [session startRunning];
    // Some cameras only take a format once running; try again if it did not stick.
    CMVideoDimensions now = CMVideoFormatDescriptionGetDimensions(dev.activeFormat.formatDescription);
    if ((now.width != gSet.width || now.height != gSet.height) && [dev lockForConfiguration:&err]) {
        dev.activeFormat = chosen;
        dev.activeVideoMinFrameDuration = CMTimeMake(1, gSet.fps);
        dev.activeVideoMaxFrameDuration = CMTimeMake(1, gSet.fps);
        [dev unlockForConfiguration];
        info.set("format_set_after_start", Value::boolean(true));
    }
    const CMVideoDimensions active = CMVideoFormatDescriptionGetDimensions(dev.activeFormat.formatDescription);
    info.set("active_format", Value::string(fmt("%dx%d, min frame duration %.4f s", active.width, active.height,
                                                CMTimeGetSeconds(dev.activeVideoMinFrameDuration))));
    info.set("session_preset", Value::string(session.sessionPreset.UTF8String));
    say(fmt("camera: %s, active format %dx%d", dev.localizedName.UTF8String, active.width, active.height));
    if (active.width != gSet.width || active.height != gSet.height)
        info.set("error", Value::string("active format differs from the requested size"));
    info.set("width", Value::integer(gSet.width));
    info.set("height", Value::integer(gSet.height));
    info.set("fps", Value::integer(gSet.fps));
    info.set("format_max_fps", Value::number(chosenMax));
    info.set("vision", Value::boolean(gSet.vision));
    info.set("share_frames", Value::boolean(gSet.shareFrames));
    gCameraCsv.open("camera.csv",
                    "seq,pts_ns,delivered_ns,vision_start_ns,vision_done_ns,face,yaw_deg,pitch_deg,roll_deg,"
                    "box_x,box_y,box_w,box_h,confidence");
    return info;  // gCameraSink.session keeps the session alive
}

// ---- iPhone reference ------------------------------------------------------

dispatch_queue_t gNetQueue;
nw_browser_t gBrowser;
nw_connection_t gRef;
dispatch_source_t gPingTimer;
uint32_t gPingSeq = 0;
std::string gRefName;

void refReceive(nw_connection_t c) {
    nw_connection_receive_message(c, ^(dispatch_data_t content, nw_content_context_t, bool, nw_error_t error) {
        const uint64_t t4 = nowNs();
        if (content) {
            const void* buf = nullptr;
            size_t len = 0;
            dispatch_data_t map = dispatch_data_create_map(content, &buf, &len);
            (void)map;
            HtHeader hdr{};
            if (len >= sizeof hdr) std::memcpy(&hdr, buf, sizeof hdr);
            if (hdr.magic == HT_MAGIC && hdr.type == HT_PONG && len == sizeof(HtPong)) {
                HtPong p;
                std::memcpy(&p, buf, sizeof p);
                gSyncCsv.row(fmt("%u,%llu,%llu,%llu,%llu", p.seq, (unsigned long long)p.t1, (unsigned long long)p.t2,
                                 (unsigned long long)p.t3, (unsigned long long)t4));
                gLastRttNs = (t4 - p.t1) - (p.t3 - p.t2);
                gPongs++;
            } else if (hdr.magic == HT_MAGIC && hdr.type == HT_POSE && len == sizeof(HtPose)) {
                HtPose p;
                std::memcpy(&p, buf, sizeof p);
                gRefCsv.row(fmt("%u,%llu,%llu,%llu,%llu,%u,%.1f,%.9f,%.9f,%.9f,%.9f,%.6f,%.6f,%.6f", p.seq,
                                (unsigned long long)p.captureNs, (unsigned long long)p.deliveredNs,
                                (unsigned long long)p.sentNs, (unsigned long long)t4, unsigned(p.tracked), p.fps, p.qx,
                                p.qy, p.qz, p.qw, p.px, p.py, p.pz));
                gRefPoses++;
                if (p.tracked) gRefTracked++;
            }
        }
        if (!error) refReceive(c);
    });
}

void refConnect(nw_endpoint_t ep) {
    nw_parameters_t params =
        nw_parameters_create_secure_udp(NW_PARAMETERS_DISABLE_PROTOCOL, NW_PARAMETERS_DEFAULT_CONFIGURATION);
    nw_connection_t c = nw_connection_create(ep, params);
    nw_connection_set_queue(c, gNetQueue);
    nw_connection_set_state_changed_handler(c, ^(nw_connection_state_t state, nw_error_t error) {
        if (state == nw_connection_state_ready) {
            gRefConnected = true;
            event("reference", "connected to " + gRefName);
        } else if (state == nw_connection_state_failed || state == nw_connection_state_cancelled) {
            if (gRef == c) {
                gRef = nil;
                gRefConnected = false;
                event("reference", state == nw_connection_state_failed ? "connection failed" : "connection closed");
            }
        }
    });
    gRef = c;
    nw_connection_start(c);
    refReceive(c);
}

void startReference() {
    gRefCsv.open("ref.csv",
                 "seq,capture_ns,delivered_ns,sent_ns,recv_ns,tracked,fps,qx,qy,qz,qw,px,py,pz");
    gSyncCsv.open("sync.csv", "seq,t1,t2,t3,t4");
    gNetQueue = dispatch_queue_create(
        "net", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
    nw_parameters_t params =
        nw_parameters_create_secure_udp(NW_PARAMETERS_DISABLE_PROTOCOL, NW_PARAMETERS_DEFAULT_CONFIGURATION);
    gBrowser = nw_browser_create(nw_browse_descriptor_create_bonjour_service(HT_BONJOUR_TYPE, "local."), params);
    nw_browser_set_queue(gBrowser, gNetQueue);
    nw_browser_set_browse_results_changed_handler(
        gBrowser, ^(nw_browse_result_t oldResult, nw_browse_result_t newResult, bool) {
            if (!newResult || gRef) return;
            nw_endpoint_t ep = nw_browse_result_copy_endpoint(newResult);
            if (!ep) return;
            const char* name = nw_endpoint_get_bonjour_service_name(ep);
            gRefName = name ? name : "?";
            event("reference", "found " + gRefName);
            refConnect(ep);
        });
    nw_browser_set_state_changed_handler(gBrowser, ^(nw_browser_state_t state, nw_error_t error) {
        if (state == nw_browser_state_failed) event("reference", "bonjour browse failed");
    });
    nw_browser_start(gBrowser);
    // Pings at 10 Hz: clock sync, and they keep the iPhone streaming to us.
    gPingTimer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, gNetQueue);
    dispatch_source_set_timer(gPingTimer, dispatch_time(DISPATCH_TIME_NOW, 0), 100 * NSEC_PER_MSEC, NSEC_PER_MSEC);
    dispatch_source_set_event_handler(gPingTimer, ^{
        if (!gRef || !gRefConnected) return;
        HtPing p{{HT_MAGIC, HT_PING}, ++gPingSeq, nowNs()};
        dispatch_data_t d = dispatch_data_create(&p, sizeof p, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        nw_connection_send(gRef, d, NW_CONNECTION_DEFAULT_MESSAGE_CONTEXT, true, ^(nw_error_t) {
        });
    });
    dispatch_resume(gPingTimer);
}

// ---- protocol --------------------------------------------------------------

struct Step {
    std::string text;
    double seconds;
    std::string target;  // laser check mark, or empty
};
std::string gProtocolName = "free";
std::vector<Step> gSteps;

struct Mark {
    std::string label;
    double x, y;  // display pixels, origin top left
};
std::vector<Mark> gMarks;
int gTargetPxW = 0, gTargetPxH = 0;
Value gTargetsJson;

bool loadProtocol(std::string& error) {
    std::ifstream in(gSet.protocolPath);
    if (!in) return error = "cannot open " + gSet.protocolPath, false;
    std::stringstream ss;
    ss << in.rdbuf();
    auto parsed = core::json::parse(ss.str());
    if (!parsed.ok) return error = parsed.error, false;
    const Value* name = parsed.value.find("name");
    const Value* steps = parsed.value.find("steps");
    if (!name || name->type() != core::json::Type::String || !steps || steps->type() != core::json::Type::Array)
        return error = "protocol needs \"name\" (string) and \"steps\" (array)", false;
    gProtocolName = name->asString();
    for (const Value& s : steps->items()) {
        const Value* t = s.find("text");
        const Value* d = s.find("seconds");
        const Value* g = s.find("target");
        if (!t || t->type() != core::json::Type::String || !d || d->type() != core::json::Type::Number ||
            d->asDouble() <= 0 || (g && g->type() != core::json::Type::String))
            return error = "each step needs \"text\" and \"seconds\" > 0 (and \"target\" is a string)", false;
        gSteps.push_back({t->asString(), d->asDouble(), g ? g->asString() : std::string()});
    }
    return true;
}

// Targets file from headtrack/analysis/laser_targets.py: geometry.screen_px and
// marks.<label>.px. Every step's target must be among the marks.
bool loadTargets(std::string& error) {
    std::ifstream in(gSet.targetsPath);
    if (!in) return error = "cannot open " + gSet.targetsPath, false;
    std::stringstream ss;
    ss << in.rdbuf();
    auto parsed = core::json::parse(ss.str());
    if (!parsed.ok) return error = parsed.error, false;
    const Value* geo = parsed.value.find("geometry");
    const Value* px = geo ? geo->find("screen_px") : nullptr;
    const Value* marks = parsed.value.find("marks");
    if (!px || px->type() != core::json::Type::Array || px->items().size() != 2 || !marks ||
        marks->type() != core::json::Type::Object)
        return error = "targets need geometry.screen_px [w, h] and marks", false;
    gTargetPxW = int(px->items()[0].asDouble());
    gTargetPxH = int(px->items()[1].asDouble());
    for (const auto& [label, m] : marks->members()) {
        const Value* p = m.find("px");
        if (!p || p->type() != core::json::Type::Array || p->items().size() != 2)
            return error = "mark " + label + " needs px [x, y]", false;
        gMarks.push_back({label, p->items()[0].asDouble(), p->items()[1].asDouble()});
    }
    for (const Step& st : gSteps)
        if (!st.target.empty() &&
            std::none_of(gMarks.begin(), gMarks.end(), [&](const Mark& m) { return m.label == st.target; }))
            return error = "protocol target " + st.target + " is not in the targets file", false;
    gTargetsJson = std::move(parsed.value);
    return true;
}

// ---- system info -----------------------------------------------------------

std::string sysctlString(const char* name) {
    char buf[256] = {0};
    size_t size = sizeof buf - 1;
    return sysctlbyname(name, buf, &size, nullptr, 0) == 0 ? buf : "";
}

Value outputDevice() {
    auto out = Value::object();
    AudioObjectPropertyAddress a{kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal,
                                 kAudioObjectPropertyElementMain};
    AudioDeviceID dev = 0;
    UInt32 size = sizeof dev;
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &a, 0, nullptr, &size, &dev) != noErr) return out;
    CFStringRef name = nullptr;
    a.mSelector = kAudioObjectPropertyName;
    size = sizeof name;
    if (AudioObjectGetPropertyData(dev, &a, 0, nullptr, &size, &name) == noErr && name) {
        out.set("name", Value::string([(__bridge NSString*)name UTF8String]));
        CFRelease(name);
    }
    UInt32 transport = 0;
    a.mSelector = kAudioDevicePropertyTransportType;
    size = sizeof transport;
    if (AudioObjectGetPropertyData(dev, &a, 0, nullptr, &size, &transport) == noErr)
        out.set("transport", Value::string(fourCC(transport)));
    return out;
}

const char* thermalName(NSProcessInfoThermalState s) {
    switch (s) {
        case NSProcessInfoThermalStateNominal: return "nominal";
        case NSProcessInfoThermalStateFair: return "fair";
        case NSProcessInfoThermalStateSerious: return "serious";
        case NSProcessInfoThermalStateCritical: return "critical";
    }
    return "unknown";
}

Value gSession = Value::object();

void writeSessionJson() {
    std::ofstream(gSet.outDir + "/session.json") << core::json::dump(gSession) << '\n';
}

}  // namespace

// ---- laser check target screen ----------------------------------------------

NSScreen* gTargetScreen = nil;  // the display the marks were computed for

NSScreen* findTargetScreen() {
    for (NSScreen* s in NSScreen.screens)
        if (std::lround(s.frame.size.width * s.backingScaleFactor) == gTargetPxW &&
            std::lround(s.frame.size.height * s.backingScaleFactor) == gTargetPxH)
            return s;
    return nil;
}

@interface TargetView : NSView
@property(nonatomic, copy) NSString* current;  // highlighted mark, or nil
@property(nonatomic, copy) NSString* prompt;
@end

@implementation TargetView

- (BOOL)isFlipped {
    return YES;  // marks are in display pixels with the origin top left
}

- (void)drawRect:(NSRect)dirty {
    [NSColor.blackColor setFill];
    NSRectFill(self.bounds);
    const CGFloat k = self.bounds.size.width / gTargetPxW;  // display pixels -> view points
    for (const Mark& m : gMarks) {
        NSString* label = [NSString stringWithUTF8String:m.label.c_str()];
        const bool on = [label isEqualToString:self.current];
        const NSPoint c = NSMakePoint(m.x * k, m.y * k);
        NSColor* col = on ? NSColor.whiteColor : [NSColor colorWithWhite:0.5 alpha:1];
        const CGFloat arm = (on ? 26 : 14) * k, r = (on ? 34 : 10) * k;
        NSBezierPath* p = [NSBezierPath bezierPath];
        [p moveToPoint:NSMakePoint(c.x - arm, c.y)];
        [p lineToPoint:NSMakePoint(c.x + arm, c.y)];
        [p moveToPoint:NSMakePoint(c.x, c.y - arm)];
        [p lineToPoint:NSMakePoint(c.x, c.y + arm)];
        [p appendBezierPathWithOvalInRect:NSMakeRect(c.x - r, c.y - r, 2 * r, 2 * r)];
        p.lineWidth = (on ? 3 : 1.5) * k;
        [col setStroke];
        [p stroke];
        NSFont* font = [NSFont systemFontOfSize:(on ? 26 : 18) * k weight:on ? NSFontWeightBold : NSFontWeightRegular];
        [label drawAtPoint:NSMakePoint(c.x + r + 4 * k, c.y + r - 6 * k)
            withAttributes:@{NSFontAttributeName : font, NSForegroundColorAttributeName : col}];
    }
    if (self.prompt.length) {
        NSMutableParagraphStyle* ps = [NSMutableParagraphStyle new];
        ps.alignment = NSTextAlignmentCenter;
        [self.prompt drawInRect:NSMakeRect(0, 30 * k, self.bounds.size.width, 60 * k)
                 withAttributes:@{NSFontAttributeName : [NSFont systemFontOfSize:30 * k weight:NSFontWeightSemibold],
                                  NSForegroundColorAttributeName : [NSColor colorWithWhite:0.8 alpha:1],
                                  NSParagraphStyleAttributeName : ps}];
    }
}

@end

// ---- window ----------------------------------------------------------------

@interface Controller : NSObject <NSWindowDelegate>
@end

@implementation Controller {
    NSWindow* _window;
    NSWindow* _targetWindow;  // laser check only
    TargetView* _targetView;
    NSTextField* _prompt;
    NSTextField* _countdown;
    NSTextField* _status;
    AVSpeechSynthesizer* _speech;
    int _step;  // -1 = waiting for Space, steps.size() = finished
    uint64_t _stepStartNs;
    uint64_t _last[4];  // airpods, camera, ref, faces at the previous status update
    uint64_t _lastStatusNs;
    uint64_t _lastPhotoNs;
    bool _ending;
}

static NSTextField* makeLabel(NSRect r, CGFloat size, NSFontWeight w) {
    NSTextField* l = [NSTextField labelWithString:@""];
    l.frame = r;
    l.font = size > 20 ? [NSFont systemFontOfSize:size weight:w] : [NSFont monospacedSystemFontOfSize:size weight:w];
    l.textColor = NSColor.whiteColor;
    l.alignment = size > 20 ? NSTextAlignmentCenter : NSTextAlignmentLeft;
    l.maximumNumberOfLines = 0;
    l.lineBreakMode = NSLineBreakByWordWrapping;
    l.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
    return l;
}

- (void)start {
    _window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 1000, 560)
                                          styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                                    NSWindowStyleMaskResizable
                                            backing:NSBackingStoreBuffered
                                              defer:NO];
    _window.title = [NSString stringWithFormat:@"HeadTrackRecorder — %s", gProtocolName.c_str()];
    _window.backgroundColor = NSColor.blackColor;
    _window.delegate = self;
    NSView* v = _window.contentView;
    _prompt = makeLabel(NSMakeRect(40, 300, 920, 200), 44, NSFontWeightSemibold);
    _countdown = makeLabel(NSMakeRect(40, 230, 920, 60), 30, NSFontWeightRegular);
    _status = makeLabel(NSMakeRect(20, 10, 960, 200), 13, NSFontWeightRegular);
    _status.textColor = NSColor.lightGrayColor;
    [v addSubview:_prompt];
    [v addSubview:_countdown];
    [v addSubview:_status];
    [_window center];
    if (gTargetScreen) [self openTargets];
    [_window makeKeyAndOrderFront:nil];
    [NSApp activate];
    _speech = [AVSpeechSynthesizer new];
    _step = -1;
    _prompt.stringValue = gSteps.empty() ? @"Recording. Press Esc to stop."
                                         : @"Put on the AirPods, face the screen.\nPress Space to start.";
    [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskKeyDown
                                          handler:^NSEvent*(NSEvent* e) {
                                              if (e.keyCode == 53) [self finish:"escape"];  // Esc
                                              else if (e.keyCode == 49 && self->_step < 0 && !gSteps.empty())
                                                  [self beginStep:0];  // Space
                                              return nil;
                                          }];
    [[NSNotificationCenter defaultCenter] addObserverForName:NSProcessInfoThermalStateDidChangeNotification
                                                      object:nil
                                                       queue:nil
                                                  usingBlock:^(NSNotification*) {
                                                      event("thermal", thermalName(NSProcessInfo.processInfo.thermalState));
                                                  }];
    [NSTimer scheduledTimerWithTimeInterval:0.1
                                    repeats:YES
                                      block:^(NSTimer*) {
                                          [self tick];
                                      }];
}

- (void)openTargets {
    const NSRect f = gTargetScreen.frame;
    _targetWindow = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, f.size.width, f.size.height)
                                                styleMask:NSWindowStyleMaskBorderless
                                                  backing:NSBackingStoreBuffered
                                                    defer:NO
                                                   screen:gTargetScreen];
    [_targetWindow setFrame:f display:NO];
    _targetWindow.level = NSStatusWindowLevel;  // above the menu bar: a plain black screen
    _targetWindow.backgroundColor = NSColor.blackColor;
    _targetWindow.releasedWhenClosed = NO;
    _targetView = [[TargetView alloc] initWithFrame:NSMakeRect(0, 0, f.size.width, f.size.height)];
    _targetView.current = @"C";
    _targetView.prompt = @"Laser on C. Press Space on the Mac to start.";
    _targetWindow.contentView = _targetView;
    [_targetWindow orderFrontRegardless];
    [NSCursor setHiddenUntilMouseMoves:YES];
    for (NSScreen* s in NSScreen.screens)  // keep the recorder window off the target screen
        if (s != gTargetScreen) {
            const NSRect vf = s.visibleFrame, wf = _window.frame;
            [_window setFrameOrigin:NSMakePoint(NSMidX(vf) - wf.size.width / 2, NSMidY(vf) - wf.size.height / 2)];
            break;
        }
}

- (void)beginStep:(int)i {
    _step = i;
    _stepStartNs = nowNs();
    if (i >= int(gSteps.size())) {
        event("protocol", "done");
        _prompt.stringValue = @"Done. Thank you.";
        _targetView.current = nil;
        _targetView.prompt = @"Done. Thank you.";
        [_targetView setNeedsDisplay:YES];
        _countdown.stringValue = @"";
        if (gSet.speak) [self speak:@"Done."];
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
            [self finish:"protocol_done"];
        });
        return;
    }
    NSString* text = [NSString stringWithUTF8String:gSteps[i].text.c_str()];
    const Step& st = gSteps[i];
    event("step", st.target.empty() ? fmt("%d/%zu %.1fs %s", i + 1, gSteps.size(), st.seconds, st.text.c_str())
                                    : fmt("%d/%zu %.1fs [%s] %s", i + 1, gSteps.size(), st.seconds, st.target.c_str(),
                                          st.text.c_str()));
    _prompt.stringValue = text;
    _targetView.current = st.target.empty() ? nil : [NSString stringWithUTF8String:st.target.c_str()];
    _targetView.prompt = text;
    [_targetView setNeedsDisplay:YES];
    if (gSet.speak) [self speak:text];
}

- (void)speak:(NSString*)text {
    [_speech stopSpeakingAtBoundary:AVSpeechBoundaryImmediate];
    [_speech speakUtterance:[AVSpeechUtterance speechUtteranceWithString:text]];
}

- (void)tick {
    const uint64_t now = nowNs();
    if (_step >= 0 && _step < int(gSteps.size())) {
        const double left = gSteps[_step].seconds - (now - _stepStartNs) / 1e9;
        if (left <= 0) [self beginStep:_step + 1];
        else _countdown.stringValue = [NSString stringWithFormat:@"%d / %zu    %.0f s", _step + 1, gSteps.size(), std::ceil(left)];
    }
    if (gSet.durationS > 0 && (now - gStartNs) / 1e9 >= gSet.durationS) [self finish:"duration"];
    if (gSet.photosS > 0 && now - _lastPhotoNs >= uint64_t(gSet.photosS * 1e9)) {
        _lastPhotoNs = now;
        [self savePhoto];
    }
    if (now - _lastStatusNs < 1'000'000'000ull) return;
    const double dt = _lastStatusNs ? (now - _lastStatusNs) / 1e9 : 1;
    const uint64_t cur[4] = {gAirpodsSamples.load(), gCameraFrames.load(), gRefPoses.load(), gVisionFaces.load()};
    const double rate[4] = {(cur[0] - _last[0]) / dt, (cur[1] - _last[1]) / dt, (cur[2] - _last[2]) / dt,
                            (cur[3] - _last[3]) / dt};
    std::memcpy(_last, cur, sizeof cur);
    _lastStatusNs = now;
    _status.stringValue = [NSString
        stringWithFormat:@"t %.0f s\nAirPods    %s %5.1f Hz   (%llu samples)\nCamera     %s %5.1f fps  (drops %llu), "
                         @"Vision face %4.1f/s\nMediaPipe  %s  (frames shared %llu)\niPhone     %s %5.1f Hz   "
                         @"(tracked %llu of %llu), last ping RTT %.1f ms, %llu pongs\nThermal    %s",
                         (now - gStartNs) / 1e9, gSet.airpods ? "on " : "off", rate[0],
                         (unsigned long long)cur[0], gSet.camera ? "on " : "off", rate[1],
                         (unsigned long long)gCameraDrops.load(), rate[3],
                         !gSet.shareFrames ? "off" : gReaderAttached ? "attached" : "not attached",
                         (unsigned long long)gFramesShared.load(),
                         !gSet.reference ? "off" : gRefConnected ? "connected" : "searching", rate[2],
                         (unsigned long long)gRefTracked.load(), (unsigned long long)cur[2], gLastRttNs.load() / 1e6,
                         (unsigned long long)gPongs.load(), thermalName(NSProcessInfo.processInfo.thermalState)];
}

// Window snapshot into <out>/photos/recorder_<UTC>.png, named by wall-clock time so it lines up with the
// webcam stills (mp_live.py --photos) and the iPhone's (HeadTrackRef --photos).
- (void)savePhoto {
    NSView* v = _window.contentView;
    NSBitmapImageRep* rep = [v bitmapImageRepForCachingDisplayInRect:v.bounds];
    NSGraphicsContext* ctx = [NSGraphicsContext graphicsContextWithBitmapImageRep:rep];
    [NSGraphicsContext saveGraphicsState];
    NSGraphicsContext.currentContext = ctx;
    [NSColor.blackColor setFill];  // the window, not the view, paints the background
    NSRectFill(v.bounds);
    [NSGraphicsContext restoreGraphicsState];
    [v displayRectIgnoringOpacity:v.bounds inContext:ctx];
    NSDateFormatter* f = [NSDateFormatter new];
    f.dateFormat = @"yyyyMMdd'T'HHmmss.SSS'Z'";
    f.timeZone = [NSTimeZone timeZoneWithAbbreviation:@"UTC"];
    f.locale = [NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"];
    NSString* dir = [NSString stringWithFormat:@"%s/photos", gSet.outDir.c_str()];
    NSString* path = [NSString stringWithFormat:@"%@/recorder_%@.png", dir, [f stringFromDate:[NSDate date]]];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        [NSFileManager.defaultManager createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
        [[rep representationUsingType:NSBitmapImageFileTypePNG properties:@{}] writeToFile:path atomically:YES];
    });
}

- (BOOL)windowShouldClose:(NSWindow*)w {
    [self finish:"window_closed"];
    return NO;
}

- (void)finish:(const char*)reason {
    if (_ending) return;
    _ending = true;
    event("end", reason);
    [gMotion stopDeviceMotionUpdates];
    [gMotionQueue waitUntilAllOperationsAreFinished];
    if (gCameraSink.session) [gCameraSink.session stopRunning];
    if (gCameraQueue) dispatch_sync(gCameraQueue, ^{
        });
    shareClose();
    if (gNetQueue) dispatch_sync(gNetQueue, ^{
            if (gRef) nw_connection_cancel(gRef);
            if (gBrowser) nw_browser_cancel(gBrowser);
        });

    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    auto end = Value::object();
    end.set("reason", Value::string(reason));
    end.set("duration_s", Value::number((nowNs() - gStartNs) / 1e9));
    end.set("airpods_samples", Value::unsignedInteger(gAirpodsSamples));
    end.set("camera_frames", Value::unsignedInteger(gCameraFrames));
    end.set("camera_drops", Value::unsignedInteger(gCameraDrops));
    end.set("vision_faces", Value::unsignedInteger(gVisionFaces));
    end.set("frames_shared", Value::unsignedInteger(gFramesShared));
    end.set("ref_poses", Value::unsignedInteger(gRefPoses));
    end.set("ref_tracked", Value::unsignedInteger(gRefTracked));
    end.set("sync_pongs", Value::unsignedInteger(gPongs));
    end.set("cpu_user_s", Value::number(ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6));
    end.set("cpu_system_s", Value::number(ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6));
    end.set("thermal", Value::string(thermalName(NSProcessInfo.processInfo.thermalState)));
    end.set("airpods_authorization", Value::integer(long(CMHeadphoneMotionManager.authorizationStatus)));
    gSession.set("end", std::move(end));
    writeSessionJson();
    say(fmt("end (%s): airpods %llu, camera %llu (drops %llu, vision faces %llu), shared %llu, ref %llu (tracked %llu), "
            "pongs %llu",
            reason, (unsigned long long)gAirpodsSamples.load(), (unsigned long long)gCameraFrames.load(),
            (unsigned long long)gCameraDrops.load(), (unsigned long long)gVisionFaces.load(),
            (unsigned long long)gFramesShared.load(), (unsigned long long)gRefPoses.load(),
            (unsigned long long)gRefTracked.load(), (unsigned long long)gPongs.load()));
    dispatch_sync(gLogQueue, ^{
        for (Csv* c : {&gAirpodsCsv, &gCameraCsv, &gRefCsv, &gSyncCsv, &gEventsCsv})
            if (c->f) std::fclose(c->f);
        if (gLogFile) std::fclose(gLogFile);
        gFilesClosed = true;
    });
    std::exit(0);
}

@end

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        mach_timebase_info(&gTimebase);
        for (int i = 1; i + 1 < argc; i += 2) {
            const std::string k = argv[i];
            const char* v = argv[i + 1];
            if (k == "--out") gSet.outDir = v;
            else if (k == "--protocol") gSet.protocolPath = v;
            else if (k == "--targets") gSet.targetsPath = v;
            else if (k == "--render-targets") gSet.renderPath = v;
            else if (k == "--duration") gSet.durationS = std::atof(v);
            else if (k == "--airpods") gSet.airpods = onOff(v);
            else if (k == "--camera") gSet.camera = onOff(v);
            else if (k == "--vision") gSet.vision = onOff(v);
            else if (k == "--share-frames") gSet.shareFrames = onOff(v);
            else if (k == "--reference") gSet.reference = onOff(v);
            else if (k == "--speak") gSet.speak = onOff(v);
            else if (k == "--width") gSet.width = std::atoi(v);
            else if (k == "--height") gSet.height = std::atoi(v);
            else if (k == "--fps") gSet.fps = std::atoi(v);
            else if (k == "--photos") gSet.photosS = std::atof(v);
            else {
                std::fprintf(stderr, "unknown option %s\n", argv[i]);
                return 1;
            }
        }
        if (!gSet.renderPath.empty()) {  // offline check of the target screen; records nothing
            std::string err;
            if (gSet.targetsPath.empty() || !loadTargets(err)) {
                std::fprintf(stderr, "targets: %s\n", gSet.targetsPath.empty() ? "--targets missing" : err.c_str());
                return 1;
            }
            TargetView* tv = [[TargetView alloc] initWithFrame:NSMakeRect(0, 0, gTargetPxW, gTargetPxH)];
            tv.current = @"R20";
            tv.prompt = @"Point at right 20 and hold";
            NSBitmapImageRep* rep = [tv bitmapImageRepForCachingDisplayInRect:tv.bounds];
            [tv cacheDisplayInRect:tv.bounds toBitmapImageRep:rep];
            NSData* png = [rep representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
            return [png writeToFile:[NSString stringWithUTF8String:gSet.renderPath.c_str()] atomically:YES] ? 0 : 1;
        }
        if (gSet.outDir.empty() || gSet.outDir[0] != '/') {
            std::fprintf(stderr,
                         "usage: HeadTrackRecorder --out <absolute dir> [--protocol file.json] [--duration s]\n"
                         "  [--airpods on|off] [--camera on|off] [--vision on|off] [--share-frames on|off]\n"
                         "  [--reference on|off] [--speak on|off] [--width 1280 --height 720 --fps 30]\n"
                         "  [--photos seconds]\n"
                         "  [--targets laser_targets.json] [--render-targets out.png]\n");
            return 1;
        }
        if (!gSet.camera) gSet.vision = gSet.shareFrames = false;
        gLogQueue = dispatch_queue_create("log", DISPATCH_QUEUE_SERIAL);
        gLogFile = std::fopen((gSet.outDir + "/log.txt").c_str(), "w");
        gEventsCsv.open("events.csv", "t_ns,kind,detail");
        if (!gSet.protocolPath.empty()) {
            std::string err;
            if (!loadProtocol(err)) {
                say("protocol: " + err);
                return 1;
            }
        }
        if (!gSet.targetsPath.empty()) {
            std::string err;
            if (!loadTargets(err)) {
                say("targets: " + err);
                return 1;
            }
        }

        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        if (!gMarks.empty() && !(gTargetScreen = findTargetScreen())) {
            say(fmt("targets: no display of %dx%d pixels connected", gTargetPxW, gTargetPxH));
            return 1;
        }
        gStartNs = nowNs();

        const NSOperatingSystemVersion os = NSProcessInfo.processInfo.operatingSystemVersion;
        gSession.set("tool", Value::string("HeadTrackRecorder"));
        gSession.set("start_utc", Value::string(core::clock::utcIsoNow()));
        gSession.set("start_ns", Value::unsignedInteger(gStartNs));
        gSession.set("clock", Value::string("mach_absolute_time in ns (CLOCK_UPTIME_RAW)"));
        gSession.set("macos", Value::string(fmt("%ld.%ld.%ld", long(os.majorVersion), long(os.minorVersion),
                                                long(os.patchVersion))));
        gSession.set("os_build", Value::string(sysctlString("kern.osversion")));
        gSession.set("model", Value::string(sysctlString("hw.model")));
        gSession.set("output_device", outputDevice());
        gSession.set("thermal_at_start", Value::string(thermalName(NSProcessInfo.processInfo.thermalState)));
        gSession.set("protocol", Value::string(gProtocolName));
        gSession.set("protocol_file", Value::string(gSet.protocolPath));
        gSession.set("speak", Value::boolean(gSet.speak));
        gSession.set("photos_s", Value::number(gSet.photosS));
        auto argsJson = Value::array();
        for (int i = 1; i < argc; ++i) argsJson.push(Value::string(argv[i]));
        gSession.set("args", std::move(argsJson));
        if (!gMarks.empty()) gSession.set("targets", gTargetsJson);
        say(fmt("session %s, protocol %s, output %s", gSet.outDir.c_str(), gProtocolName.c_str(),
                core::json::dump(gSession.find("output_device") ? *gSession.find("output_device") : Value::object(), 0)
                    .c_str()));

        if (gSet.airpods) startAirpods();
        if (gSet.camera) {
            Value cam = startCamera();
            if (const Value* e = cam.find("error")) say("camera: " + e->asString());
            gSession.set("camera", std::move(cam));
        }
        if (gSet.reference) startReference();
        writeSessionJson();

        Controller* controller = [Controller new];
        [controller start];
        [NSApp run];
    }
    return 0;
}
