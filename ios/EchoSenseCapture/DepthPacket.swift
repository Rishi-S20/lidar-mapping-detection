import CoreMotion
import CoreVideo
import Foundation
import simd

/// One frame of the depth stream v0.1 (see PROTOCOL.md).
struct DepthPacket {
    var seq: Int
    var t: Double
    var source: String          // "lidar" or "arkit"
    var width: Int
    var height: Int
    var fx: Float, fy: Float, cx: Float, cy: Float
    var depth16: Data           // width*height Float16 meters, 0 = no return
    var confidence: Data?       // width*height UInt8 (0 low, 1 medium, 2 high)
    var filtered: Bool
    var motion: CMDeviceMotion?
    var arkitPose: simd_float4x4?
    var arkitTracking: String?
    var ambientLux: Double?
    var attitudeRef: String = "arbitrary"

    func encoded() -> Data {
        var header: [String: Any] = [
            "v": "0.1", "seq": seq, "t": t, "source": source, "w": width, "h": height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "confidence": confidence != nil, "filtered": filtered, "attitude_ref": attitudeRef,
        ]
        if let m = motion {
            let q = m.attitude.quaternion
            header["attitude"] = [q.w, q.x, q.y, q.z]
            header["gravity"] = [m.gravity.x, m.gravity.y, m.gravity.z]
            header["rotation_rate"] = [m.rotationRate.x, m.rotationRate.y, m.rotationRate.z]
            header["user_accel"] = [m.userAcceleration.x, m.userAcceleration.y, m.userAcceleration.z]
        } else {
            header["attitude"] = [1.0, 0.0, 0.0, 0.0]
            header["gravity"] = [0.0, 0.0, 0.0]
        }
        if let p = arkitPose {
            header["arkit_pose"] = [p.columns.0, p.columns.1, p.columns.2, p.columns.3]
                .flatMap { [$0.x, $0.y, $0.z, $0.w] }
        }
        if let s = arkitTracking { header["arkit_tracking"] = s }
        if let l = ambientLux { header["ambient_lux"] = l }

        let json = (try? JSONSerialization.data(withJSONObject: header)) ?? Data()
        var out = Data(capacity: 4 + json.count + depth16.count + (confidence?.count ?? 0))
        var len = UInt32(json.count).littleEndian
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(json)
        out.append(depth16)
        if let c = confidence { out.append(c) }
        return out
    }
}

enum PixelBufferCopy {
    /// Depth pixel buffer (Float32 or Float16 meters) -> tightly packed Float16 data, plus the valid fraction.
    static func float16(from pb: CVPixelBuffer) -> (data: Data, width: Int, height: Int, valid: Double)? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        let fmt = CVPixelBufferGetPixelFormatType(pb)
        var out = Data(count: w * h * 2)
        var valid = 0
        out.withUnsafeMutableBytes { raw in
            let dst = raw.bindMemory(to: Float16.self)
            for y in 0..<h {
                let row = base.advanced(by: y * bpr)
                for x in 0..<w {
                    let v: Float
                    switch fmt {
                    case kCVPixelFormatType_DepthFloat32, kCVPixelFormatType_DisparityFloat32:
                        v = row.assumingMemoryBound(to: Float32.self)[x]
                    case kCVPixelFormatType_DepthFloat16, kCVPixelFormatType_DisparityFloat16:
                        v = Float(row.assumingMemoryBound(to: Float16.self)[x])
                    default:
                        v = 0
                    }
                    if v.isFinite && v > 0 {
                        dst[y * w + x] = Float16(v)
                        valid += 1
                    } else {
                        dst[y * w + x] = 0
                    }
                }
            }
        }
        return (out, w, h, Double(valid) / Double(max(w * h, 1)))
    }

    /// One-byte-per-pixel buffer (ARKit confidence map) -> tightly packed data.
    static func bytes(from pb: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        var out = Data(count: w * h)
        out.withUnsafeMutableBytes { raw in
            let dst = raw.bindMemory(to: UInt8.self)
            for y in 0..<h {
                let row = base.advanced(by: y * bpr).assumingMemoryBound(to: UInt8.self)
                for x in 0..<w { dst[y * w + x] = row[x] }
            }
        }
        return out
    }
}
