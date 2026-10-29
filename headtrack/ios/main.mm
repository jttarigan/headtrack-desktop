// HeadTrackRef: the side paper's reference tracker (HEADTRACK_STUDY.md).
// Runs ARKit face tracking with the TrueDepth camera on the iPhone and streams
// one head pose per camera frame to HeadTrackRecorder on the Mac over UDP.
// Also answers the Mac's clock-sync pings. Advertised by Bonjour, so no
// addresses need typing. The Mac pings at 10 Hz; poses go to whoever pinged
// last, and stop 2 s after the pings stop.
// Launch argument --photos N saves the screen every N s as
// Documents/photos/ref_<UTC>.jpg, for figures of the test process.

#import <ARKit/ARKit.h>
#import <Network/Network.h>
#import <UIKit/UIKit.h>
#include <mach/mach_time.h>

#include <atomic>
#include <cmath>
#include <cstring>

#include "common/wire.h"

namespace {

mach_timebase_info_data_t gTimebase;

uint64_t nowNs() { return mach_absolute_time() * gTimebase.numer / gTimebase.denom; }
uint64_t secondsToNs(double s) { return uint64_t(s * 1e9 + 0.5); }

// State shared between the ARKit queue, the network queue and the UI timer.
dispatch_queue_t gNetQueue;
nw_connection_t gPeer;                 // last connection that pinged; touched on gNetQueue only
std::atomic<uint64_t> gLastPingNs{0};
std::atomic<uint32_t> gPings{0}, gPosesSent{0}, gFrames{0}, gTrackedFrames{0};
std::atomic<float> gYaw{0}, gPitch{0}, gRoll{0};
std::atomic<bool> gTracked{false};

void send(nw_connection_t c, const void* bytes, size_t n) {
    dispatch_data_t d = dispatch_data_create(bytes, n, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    nw_connection_send(c, d, NW_CONNECTION_DEFAULT_MESSAGE_CONTEXT, true, ^(nw_error_t) {
    });
}

void receiveLoop(nw_connection_t c) {
    nw_connection_receive_message(c, ^(dispatch_data_t content, nw_content_context_t, bool, nw_error_t error) {
        const uint64_t t2 = nowNs();
        if (content) {
            const void* buf = nullptr;
            size_t len = 0;
            dispatch_data_t map = dispatch_data_create_map(content, &buf, &len);
            (void)map;
            HtPing ping;
            if (len == sizeof ping) {
                std::memcpy(&ping, buf, sizeof ping);
                if (ping.h.magic == HT_MAGIC && ping.h.type == HT_PING) {
                    HtPong pong{{HT_MAGIC, HT_PONG}, ping.seq, ping.t1, t2, 0};
                    pong.t3 = nowNs();
                    send(c, &pong, sizeof pong);
                    gPeer = c;
                    gLastPingNs = t2;
                    gPings++;
                }
            }
        }
        if (!error) receiveLoop(c);
    });
}

void startListener() {
    nw_parameters_t params =
        nw_parameters_create_secure_udp(NW_PARAMETERS_DISABLE_PROTOCOL, NW_PARAMETERS_DEFAULT_CONFIGURATION);
    nw_listener_t listener = nw_listener_create(params);
    nw_listener_set_advertise_descriptor(listener,
                                         nw_advertise_descriptor_create_bonjour_service("HeadTrackRef", HT_BONJOUR_TYPE, nullptr));
    nw_listener_set_queue(listener, gNetQueue);
    nw_listener_set_new_connection_handler(listener, ^(nw_connection_t c) {
        nw_connection_set_queue(c, gNetQueue);
        nw_connection_start(c);
        receiveLoop(c);
    });
    nw_listener_start(listener);
}

}  // namespace

@interface ViewController : UIViewController <ARSessionDelegate>
@end

@implementation ViewController {
    ARSCNView* _view;
    UILabel* _status;
    dispatch_queue_t _arQueue;
    float _fps;
    NSString* _formats;
    double _photosS;  // 0 = no photos
    uint64_t _lastPhotoNs;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    _view = [[ARSCNView alloc] initWithFrame:self.view.bounds];
    _view.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    [self.view addSubview:_view];

    _status = [[UILabel alloc] initWithFrame:CGRectInset(self.view.bounds, 16, 60)];
    _status.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    _status.numberOfLines = 0;
    _status.textColor = UIColor.whiteColor;
    _status.shadowColor = UIColor.blackColor;
    _status.font = [UIFont monospacedSystemFontOfSize:15 weight:UIFontWeightSemibold];
    _status.textAlignment = NSTextAlignmentLeft;
    [self.view addSubview:_status];

    NSArray<NSString*>* args = NSProcessInfo.processInfo.arguments;
    const NSUInteger i = [args indexOfObject:@"--photos"];
    if (i != NSNotFound && i + 1 < args.count) _photosS = args[i + 1].doubleValue;

    _arQueue = dispatch_queue_create("ar", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL,
                                                                                   QOS_CLASS_USER_INTERACTIVE, 0));
    [NSTimer scheduledTimerWithTimeInterval:0.25
                                    repeats:YES
                                      block:^(NSTimer*) {
                                          [self updateStatus];
                                      }];
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    UIApplication.sharedApplication.idleTimerDisabled = YES;
    if (!ARFaceTrackingConfiguration.isSupported) {
        _status.text = @"ARKit face tracking is not supported on this device.";
        return;
    }
    ARFaceTrackingConfiguration* config = [ARFaceTrackingConfiguration new];
    config.maximumNumberOfTrackedFaces = 1;
    config.worldTrackingEnabled = NO;
    config.lightEstimationEnabled = NO;
    // Highest frame rate available; ties go to the first (ARKit's preferred) format.
    ARVideoFormat* best = nil;
    NSMutableString* formats = [NSMutableString string];
    for (ARVideoFormat* f in ARFaceTrackingConfiguration.supportedVideoFormats) {
        [formats appendFormat:@"%.0fx%.0f@%ld ", f.imageResolution.width, f.imageResolution.height,
                              long(f.framesPerSecond)];
        if (!best || f.framesPerSecond > best.framesPerSecond) best = f;
    }
    _formats = formats;
    if (best) config.videoFormat = best;
    _fps = best ? float(best.framesPerSecond) : 0;
    _view.session.delegate = self;
    _view.session.delegateQueue = _arQueue;
    [_view.session runWithConfiguration:config];
}

- (void)session:(ARSession*)session didUpdateFrame:(ARFrame*)frame {
    const uint64_t delivered = nowNs();
    HtPose p{};
    p.h = {HT_MAGIC, HT_POSE};
    p.seq = gFrames++;
    p.captureNs = secondsToNs(frame.timestamp);
    p.deliveredNs = delivered;
    p.fps = _fps;
    p.qw = 1;
    for (ARAnchor* a in frame.anchors) {
        if (![a isKindOfClass:ARFaceAnchor.class]) continue;
        ARFaceAnchor* face = (ARFaceAnchor*)a;
        if (!face.isTracked) break;
        const simd_float4x4 m = simd_mul(simd_inverse(frame.camera.transform), face.transform);
        const simd_quatf q = simd_quaternion(m);
        p.tracked = 1;
        p.qx = q.vector.x, p.qy = q.vector.y, p.qz = q.vector.z, p.qw = q.vector.w;
        p.px = m.columns[3].x, p.py = m.columns[3].y, p.pz = m.columns[3].z;
        // Display only (the Mac gets the quaternion): rough Euler angles in camera axes.
        const simd_float3 fwd = simd_make_float3(m.columns[2].x, m.columns[2].y, m.columns[2].z);
        gYaw = std::atan2(fwd.x, fwd.z) * 180 / M_PI;
        gPitch = std::asin(std::fmax(-1.f, std::fmin(1.f, fwd.y))) * 180 / M_PI;
        gRoll = std::atan2(m.columns[0].y, m.columns[1].y) * 180 / M_PI;
        break;
    }
    gTracked = p.tracked;
    if (p.tracked) gTrackedFrames++;
    dispatch_async(gNetQueue, ^{
        if (!gPeer || nowNs() - gLastPingNs.load() > 2'000'000'000ull) return;
        HtPose out = p;
        out.sentNs = nowNs();
        send(gPeer, &out, sizeof out);
        gPosesSent++;
    });
}

- (void)updateStatus {
    const double sincePing = gLastPingNs.load() ? (nowNs() - gLastPingNs.load()) / 1e9 : -1;
    NSString* mac = sincePing < 0 ? @"waiting for Mac" : sincePing < 2 ? @"connected" : @"lost";
    _status.text = [NSString
        stringWithFormat:@"HeadTrackRef\n\nFace: %@\nyaw %+6.1f  pitch %+6.1f  roll %+6.1f\n\nCamera: %.0f fps\nFrames: %u (tracked %u)\n\nMac: %@\npings %u, poses sent %u\n\nFormats: %@",
                         gTracked.load() ? @"TRACKED" : @"not found", gYaw.load(), gPitch.load(), gRoll.load(), _fps,
                         gFrames.load(), gTrackedFrames.load(), mac, gPings.load(), gPosesSent.load(), _formats ?: @""];
    if (_photosS > 0 && nowNs() - _lastPhotoNs >= secondsToNs(_photosS)) {
        _lastPhotoNs = nowNs();
        [self savePhoto];
    }
}

// What the screen shows: the ARKit camera view with the status text drawn on top.
- (void)savePhoto {
    UIImage* scene = [_view snapshot];
    const CGRect b = self.view.bounds;
    UIImage* shot = [[[UIGraphicsImageRenderer alloc] initWithSize:b.size]
        imageWithActions:^(UIGraphicsImageRendererContext* ctx) {
            [scene drawInRect:b];
            CGContextTranslateCTM(ctx.CGContext, self->_status.frame.origin.x, self->_status.frame.origin.y);
            [self->_status.layer renderInContext:ctx.CGContext];
        }];
    NSDateFormatter* f = [NSDateFormatter new];
    f.dateFormat = @"yyyyMMdd'T'HHmmss.SSS'Z'";
    f.timeZone = [NSTimeZone timeZoneWithAbbreviation:@"UTC"];
    f.locale = [NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"];
    NSString* dir = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject
        stringByAppendingPathComponent:@"photos"];
    NSString* path = [dir stringByAppendingFormat:@"/ref_%@.jpg", [f stringFromDate:[NSDate date]]];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        [NSFileManager.defaultManager createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
        [UIImageJPEGRepresentation(shot, 0.9) writeToFile:path atomically:YES];
    });
}

- (BOOL)prefersStatusBarHidden {
    return YES;
}

@end

@interface SceneDelegate : UIResponder <UIWindowSceneDelegate>
@property(strong, nonatomic) UIWindow* window;
@end

@implementation SceneDelegate
- (void)scene:(UIScene*)scene willConnectToSession:(UISceneSession*)session options:(UISceneConnectionOptions*)options {
    self.window = [[UIWindow alloc] initWithWindowScene:(UIWindowScene*)scene];
    self.window.rootViewController = [ViewController new];
    [self.window makeKeyAndVisible];
}
@end

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@end

@implementation AppDelegate
- (BOOL)application:(UIApplication*)app didFinishLaunchingWithOptions:(NSDictionary*)options {
    mach_timebase_info(&gTimebase);
    gNetQueue = dispatch_queue_create("net", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL,
                                                                                     QOS_CLASS_USER_INTERACTIVE, 0));
    startListener();
    return YES;
}
- (UISceneConfiguration*)application:(UIApplication*)app
    configurationForConnectingSceneSession:(UISceneSession*)session
                                   options:(UISceneConnectionOptions*)options {
    UISceneConfiguration* c = [[UISceneConfiguration alloc] initWithName:@"Default" sessionRole:session.role];
    c.delegateClass = SceneDelegate.class;
    return c;
}
@end

int main(int argc, char* argv[]) {
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass(AppDelegate.class));
    }
}
