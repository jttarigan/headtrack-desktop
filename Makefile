# Head-tracker characterisation: Mac recorder and iPhone reference app. Needs Xcode's command-line tools.
#
#   make recorder                      build build/HeadTrackRecorder.app (ad-hoc signed unless DEV_IDENTITY is set)
#   make ios DEV_IDENTITY="Apple Development: ..." IOS_PROFILE=/path/to/profile.mobileprovision
#                                      build build/ios/HeadTrackRef.app (needs your own team; change the bundle id
#                                      in headtrack/ios/Info.plist and the team in headtrack/ios/entitlements.plist)
#   make clean

BUILD        := build
DEV_IDENTITY ?= -
IOS_PROFILE  ?=
CXX          := xcrun clang++
IOS_CXX      := xcrun -sdk iphoneos clang++
ARCHFLAGS    := -arch arm64 -mmacosx-version-min=14.0
WARN         := -Wall -Wextra -Wshadow -Wno-unused-parameter
OPT          := -O2 -g -DNDEBUG
OBJCXXFLAGS  := -std=c++20 $(ARCHFLAGS) $(OPT) $(WARN) -Isrc -Iheadtrack -fobjc-arc -MMD -MP

REC_APP  := $(BUILD)/HeadTrackRecorder.app
REC_BIN  := $(REC_APP)/Contents/MacOS/HeadTrackRecorder
REC_OBJS := $(BUILD)/obj/recorder.o $(BUILD)/obj/clock.o $(BUILD)/obj/json.o
IOS_APP  := $(BUILD)/ios/HeadTrackRef.app

.PHONY: recorder ios clean

recorder: $(REC_BIN) $(REC_APP)/Contents/Info.plist
	codesign --force --sign "$(DEV_IDENTITY)" --timestamp=none $(REC_APP)

$(REC_BIN): $(REC_OBJS)
	@mkdir -p $(@D)
	$(CXX) $(ARCHFLAGS) -framework Cocoa -framework AVFoundation -framework CoreMedia -framework CoreVideo \
	  -framework CoreMotion -framework Network -framework Vision -framework CoreAudio $^ -o $@

$(REC_APP)/Contents/Info.plist: headtrack/mac/Info.plist
	@mkdir -p $(@D)
	cp $< $@

$(BUILD)/obj/recorder.o: headtrack/mac/recorder.mm
	@mkdir -p $(@D)
	$(CXX) $(OBJCXXFLAGS) -c $< -o $@

$(BUILD)/obj/%.o: src/core/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(OBJCXXFLAGS) -c $< -o $@

ios: headtrack/ios/main.mm headtrack/common/wire.h headtrack/ios/Info.plist headtrack/ios/entitlements.plist
	@test -n "$(IOS_PROFILE)" || { echo "set IOS_PROFILE and DEV_IDENTITY (see the top of this Makefile)"; exit 1; }
	@mkdir -p $(IOS_APP)
	$(IOS_CXX) -std=c++20 -arch arm64 -miphoneos-version-min=17.0 $(OPT) $(WARN) -fobjc-arc -Iheadtrack \
	  headtrack/ios/main.mm -framework UIKit -framework ARKit -framework SceneKit -framework Network -framework CoreGraphics \
	  -o $(IOS_APP)/HeadTrackRef
	cp headtrack/ios/Info.plist $(IOS_APP)/Info.plist
	cp "$(IOS_PROFILE)" $(IOS_APP)/embedded.mobileprovision
	codesign --force --sign "$(DEV_IDENTITY)" --entitlements headtrack/ios/entitlements.plist --timestamp=none $(IOS_APP)

clean:
	rm -rf $(BUILD)

-include $(REC_OBJS:.o=.d)
