import SwiftUI

struct ContentView: View {
    @StateObject private var capture = CaptureController()
    @AppStorage("macURL") private var macURL = "ws://172.20.10.2:8770/depth"
    @AppStorage("fps") private var fps = 10.0
    @AppStorage("filtered") private var filtered = false
    @State private var source: DepthSourceKind = .lidar

    var body: some View {
        // Dark UI on purpose: this is used in the dark and shouldn't light up the room.
        NavigationStack {
            Form {
                Section("Mac mapper") {
                    TextField("ws://<mac-ip>:8770/depth", text: $macURL)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .disabled(capture.running)
                    Text("Run `lidar_mapper serve` on the Mac. Over the iPhone's USB hotspot the Mac is usually 172.20.10.2.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                Section("Capture") {
                    Picker("Depth source", selection: $source) {
                        ForEach(DepthSourceKind.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .disabled(capture.running)
                    if source == .arkit {
                        Text("ARKit's pose uses the camera and fails in the dark. Use it only to compare against LiDAR-only mapping with the lights on.")
                            .font(.footnote).foregroundStyle(.orange)
                    }
                    Toggle("Hole filling (AVFoundation)", isOn: $filtered).disabled(capture.running || source != .lidar)
                    Stepper("Frames per second: \(Int(fps))", value: $fps, in: 2...30, step: 1).disabled(capture.running)
                }
                Section {
                    Button(capture.running ? "Stop" : "Start streaming") {
                        if capture.running { capture.stop() }
                        else { capture.start(urlString: macURL, source: source, targetFPS: fps, filtered: filtered) }
                    }
                    .font(.headline)
                    .foregroundStyle(capture.running ? .red : .green)
                }
                Section("Status") {
                    row("Connection", capture.connection)
                    row("Depth map", capture.depthSize)
                    row("Valid depth", String(format: "%.0f%%", capture.validDepth * 100))
                    row("Sent / dropped", "\(capture.sent) / \(capture.dropped)")
                    row("Rate", String(format: "%.1f fps", capture.fps))
                    row("Pose", capture.tracking)
                    if let lux = capture.lux { row("Light (ARKit)", String(format: "%.0f", lux)) }
                    if let e = capture.error { Text(e).foregroundStyle(.red).font(.footnote) }
                }
            }
            .navigationTitle("LiDAR Capture")
        }
        .preferredColorScheme(.dark)
    }

    private func row(_ k: String, _ v: String) -> some View {
        HStack { Text(k); Spacer(); Text(v).foregroundStyle(.secondary).monospacedDigit() }
    }
}
