import ARKit
import AVFoundation
import Foundation
import UIKit

enum DepthSourceKind: String, CaseIterable, Identifiable {
    case lidar = "LiDAR only"
    case arkit = "ARKit (reference)"
    var id: String { rawValue }
}

/// Ties together depth source, IMU, and the WebSocket. Publishes stats for the UI.
@MainActor
final class CaptureController: ObservableObject {
    @Published var running = false
    @Published var connection = "idle"
    @Published var sent = 0
    @Published var dropped = 0
    @Published var fps = 0.0
    @Published var validDepth = 0.0
    @Published var depthSize = "-"
    @Published var tracking = "-"
    @Published var lux: Double?
    @Published var error: String?

    nonisolated private let motion = MotionTracker()
    nonisolated private let streamer = FrameStreamer()
    private var lidar: LiDARDepthSource?
    private var arkit: ARKitDepthSource?

    // touched only on the capture queues
    nonisolated(unsafe) private var seq = 0
    nonisolated(unsafe) private var lastSent = 0.0
    nonisolated(unsafe) private var minInterval = 0.1
    nonisolated(unsafe) private var filtered = false
    nonisolated(unsafe) private var fpsEMA = 0.0
    nonisolated(unsafe) private var lastUIUpdate = 0.0
    nonisolated(unsafe) private var sentCount = 0
    nonisolated(unsafe) private var droppedCount = 0

    init() {
        streamer.onState = { [weak self] s in
            Task { @MainActor in
                self?.connection = s.rawValue
                if s == .failed { self?.error = self?.streamer.lastError }
                if s == .connected { self?.error = nil }
            }
        }
    }

    func start(urlString: String, source: DepthSourceKind, targetFPS: Double, filtered: Bool) {
        guard let url = URL(string: urlString), url.scheme == "ws" || url.scheme == "wss" else {
            error = "Enter a ws:// address, e.g. ws://172.20.10.2:8770/depth"
            return
        }
        error = nil
        seq = 0; sentCount = 0; droppedCount = 0; fpsEMA = 0
        minInterval = 1.0 / max(1, min(30, targetFPS))
        self.filtered = filtered
        motion.start()
        streamer.connect(url)
        do {
            switch source {
            case .lidar:
                let s = LiDARDepthSource()
                s.onDepth = { [weak self] d, t in self?.handle(depth: d, time: t) }
                try s.start(filtered: filtered)
                lidar = s
            case .arkit:
                let s = ARKitDepthSource()
                s.onFrame = { [weak self] f in self?.handle(frame: f) }
                try s.start()
                arkit = s
            }
            running = true
            UIApplication.shared.isIdleTimerDisabled = true
        } catch {
            self.error = error.localizedDescription
            stop()
        }
    }

    func stop() {
        lidar?.stop(); lidar = nil
        arkit?.stop(); arkit = nil
        motion.stop()
        streamer.disconnect()
        running = false
        UIApplication.shared.isIdleTimerDisabled = false
    }

    // MARK: - capture callbacks (background queues)

    nonisolated private func shouldSend(_ t: Double) -> Bool {
        if t - lastSent < minInterval { return false }
        if lastSent > 0 { fpsEMA = fpsEMA == 0 ? 1 / (t - lastSent) : 0.8 * fpsEMA + 0.2 / (t - lastSent) }
        lastSent = t
        return true
    }

    nonisolated private func handle(depth: AVDepthData, time: Double) {
        guard shouldSend(time) else { return }
        let d = depth.depthDataType == kCVPixelFormatType_DepthFloat16 || depth.depthDataType == kCVPixelFormatType_DepthFloat32
            ? depth : depth.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        guard let cal = d.cameraCalibrationData,
              let copy = PixelBufferCopy.float16(from: d.depthDataMap) else {
            report(valid: 0, size: "-", tracking: "no calibration", lux: nil); return
        }
        let ref = cal.intrinsicMatrixReferenceDimensions
        let sx = Float(copy.width) / Float(ref.width), sy = Float(copy.height) / Float(ref.height)
        let K = cal.intrinsicMatrix  // column-major: [0][0]=fx, [1][1]=fy, [2][0]=cx, [2][1]=cy
        let packet = DepthPacket(
            seq: nextSeq(), t: time, source: "lidar", width: copy.width, height: copy.height,
            fx: K[0][0] * sx, fy: K[1][1] * sy, cx: K[2][0] * sx, cy: K[2][1] * sy,
            depth16: copy.data, confidence: nil, filtered: filtered, motion: motion.sample(),
            arkitPose: nil, arkitTracking: nil, ambientLux: nil, attitudeRef: motion.referenceName)
        send(packet, valid: copy.valid, tracking: "LiDAR only (no camera tracking)", lux: nil)
    }

    nonisolated private func handle(frame: ARFrame) {
        guard shouldSend(frame.timestamp), let sd = frame.sceneDepth,
              let copy = PixelBufferCopy.float16(from: sd.depthMap) else { return }
        let conf = sd.confidenceMap.flatMap { PixelBufferCopy.bytes(from: $0) }
        let res = frame.camera.imageResolution
        let sx = Float(copy.width) / Float(res.width), sy = Float(copy.height) / Float(res.height)
        let K = frame.camera.intrinsics
        let lux = frame.lightEstimate.map { Double($0.ambientIntensity) }
        let state = frame.camera.trackingState.label
        let packet = DepthPacket(
            seq: nextSeq(), t: frame.timestamp, source: "arkit", width: copy.width, height: copy.height,
            fx: K[0][0] * sx, fy: K[1][1] * sy, cx: K[2][0] * sx, cy: K[2][1] * sy,
            depth16: copy.data, confidence: conf, filtered: false, motion: motion.sample(),
            arkitPose: frame.camera.transform, arkitTracking: state, ambientLux: lux,
            attitudeRef: motion.referenceName)
        send(packet, valid: copy.valid, tracking: "ARKit: \(state)", lux: lux)
    }

    nonisolated private func nextSeq() -> Int { seq += 1; return seq - 1 }

    nonisolated private func send(_ p: DepthPacket, valid: Double, tracking: String, lux: Double?) {
        if streamer.send(p.encoded()) { sentCount += 1 } else { droppedCount += 1 }
        report(valid: valid, size: "\(p.width)x\(p.height)", tracking: tracking, lux: lux)
    }

    nonisolated private func report(valid: Double, size: String, tracking: String, lux: Double?) {
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastUIUpdate > 0.25 else { return }
        lastUIUpdate = now
        let (s, d, f) = (sentCount, droppedCount, fpsEMA)
        Task { @MainActor in
            self.sent = s; self.dropped = d; self.fps = f
            self.validDepth = valid; self.depthSize = size; self.tracking = tracking; self.lux = lux
        }
    }
}
