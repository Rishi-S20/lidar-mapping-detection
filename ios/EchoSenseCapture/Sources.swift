import ARKit
import AVFoundation
import CoreMotion
import Foundation

enum CaptureError: LocalizedError {
    case noLiDAR, noDepthFormat, cannotAdd(String), arkitDepthUnsupported

    var errorDescription: String? {
        switch self {
        case .noLiDAR: return "This iPhone has no LiDAR depth camera."
        case .noDepthFormat: return "No LiDAR depth format available."
        case .cannotAdd(let what): return "Could not add \(what) to the capture session."
        case .arkitDepthUnsupported: return "ARKit scene depth is not supported on this device."
        }
    }
}

/// IMU orientation: gyro + accelerometer + magnetometer. Needs no light.
final class MotionTracker: @unchecked Sendable {
    private let manager = CMMotionManager()
    private let queue = OperationQueue()
    private let lock = NSLock()
    private var latest: CMDeviceMotion?
    /// "magnetic_north" keeps the same heading across app restarts, so the Mac only has to find the
    /// phone's position when a new session starts. The arbitrary frames reset yaw on every start.
    private(set) var referenceName = "arbitrary"

    func start() {
        guard manager.isDeviceMotionAvailable else { return }
        manager.deviceMotionUpdateInterval = 1.0 / 100.0
        let frames = CMMotionManager.availableAttitudeReferenceFrames()
        let ref: CMAttitudeReferenceFrame
        if frames.contains(.xMagneticNorthZVertical) {
            ref = .xMagneticNorthZVertical; referenceName = "magnetic_north"
        } else if frames.contains(.xArbitraryCorrectedZVertical) {
            ref = .xArbitraryCorrectedZVertical; referenceName = "arbitrary"
        } else {
            ref = .xArbitraryZVertical; referenceName = "arbitrary"
        }
        manager.startDeviceMotionUpdates(using: ref, to: queue) { [weak self] motion, _ in
            guard let self, let motion else { return }
            self.lock.lock(); self.latest = motion; self.lock.unlock()
        }
    }

    func stop() { manager.stopDeviceMotionUpdates() }

    func sample() -> CMDeviceMotion? {
        lock.lock(); defer { lock.unlock() }
        return latest
    }
}

/// LiDAR depth through AVFoundation. No world tracking, so nothing here depends on visible light.
final class LiDARDepthSource: NSObject, AVCaptureDepthDataOutputDelegate {
    let session = AVCaptureSession()
    private let depthOutput = AVCaptureDepthDataOutput()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "echosense.depth")
    var onDepth: ((AVDepthData, Double) -> Void)?

    func start(filtered: Bool) throws {
        guard let device = AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back) else {
            throw CaptureError.noLiDAR
        }
        session.beginConfiguration()
        session.sessionPreset = .inputPriority
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CaptureError.cannotAdd("LiDAR input") }
        session.addInput(input)

        // A video output keeps the depth pipeline running; its frames are discarded, never sent.
        videoOutput.alwaysDiscardsLateVideoFrames = true
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }

        depthOutput.isFilteringEnabled = filtered
        depthOutput.alwaysDiscardsLateDepthData = true
        guard session.canAddOutput(depthOutput) else { throw CaptureError.cannotAdd("depth output") }
        session.addOutput(depthOutput)
        depthOutput.setDelegate(self, callbackQueue: queue)

        func isDepth(_ f: AVCaptureDevice.Format) -> Bool {
            let t = CMFormatDescriptionGetMediaSubType(f.formatDescription)
            return t == kCVPixelFormatType_DepthFloat16 || t == kCVPixelFormatType_DepthFloat32
        }
        func width(_ f: AVCaptureDevice.Format) -> Int32 {
            CMVideoFormatDescriptionGetDimensions(f.formatDescription).width
        }
        // Smallest video format that still offers the largest depth map: least work for the phone.
        let candidates = device.formats.filter { !$0.supportedDepthDataFormats.filter(isDepth).isEmpty }
        let bestDepth = candidates.map { $0.supportedDepthDataFormats.filter(isDepth).map(width).max() ?? 0 }.max() ?? 0
        guard let format = candidates
            .filter({ ($0.supportedDepthDataFormats.filter(isDepth).map(width).max() ?? 0) == bestDepth })
            .min(by: { width($0) < width($1) }),
              let depthFormat = format.supportedDepthDataFormats.filter(isDepth).max(by: { width($0) < width($1) })
        else { throw CaptureError.noDepthFormat }
        try device.lockForConfiguration()
        device.activeFormat = format
        device.activeDepthDataFormat = depthFormat
        device.unlockForConfiguration()
        session.commitConfiguration()
        queue.async { self.session.startRunning() }
    }

    func stop() { queue.async { self.session.stopRunning() } }

    func depthDataOutput(_ output: AVCaptureDepthDataOutput, didOutput depthData: AVDepthData,
                         timestamp: CMTime, connection: AVCaptureConnection) {
        onDepth?(depthData, timestamp.seconds)
    }
}

/// ARKit scene depth. Optional comparison mode: its pose is camera-based and is marked reference-only.
final class ARKitDepthSource: NSObject, ARSessionDelegate {
    let session = ARSession()
    private let queue = DispatchQueue(label: "echosense.arkit")
    var onFrame: ((ARFrame) -> Void)?

    func start() throws {
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            throw CaptureError.arkitDepthUnsupported
        }
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth]
        config.isLightEstimationEnabled = true
        session.delegate = self
        session.delegateQueue = queue
        session.run(config)
    }

    func stop() { session.pause() }

    func session(_ session: ARSession, didUpdate frame: ARFrame) { onFrame?(frame) }
}

extension ARCamera.TrackingState {
    var label: String {
        switch self {
        case .normal: return "normal"
        case .notAvailable: return "notAvailable"
        case .limited(let r):
            switch r {
            case .initializing: return "limited:initializing"
            case .excessiveMotion: return "limited:excessiveMotion"
            case .insufficientFeatures: return "limited:insufficientFeatures"
            case .relocalizing: return "limited:relocalizing"
            @unknown default: return "limited:unknown"
            }
        }
    }
}
