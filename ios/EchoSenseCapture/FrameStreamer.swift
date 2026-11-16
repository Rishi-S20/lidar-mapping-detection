import Foundation

/// WebSocket client to the Mac mapper. At most one frame in flight: if the network is
/// slow, new frames are dropped instead of queued (the Mac only wants the newest).
final class FrameStreamer: NSObject, URLSessionWebSocketDelegate, @unchecked Sendable {
    enum State: String { case idle, connecting, connected, failed }

    private var session: URLSession!
    private var task: URLSessionWebSocketTask?
    private let queue = DispatchQueue(label: "echosense.stream")
    private var sending = false
    private var url: URL?
    private(set) var state: State = .idle { didSet { onState?(state) } }
    var onState: ((State) -> Void)?
    var lastError: String?

    override init() {
        super.init()
        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = false
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    func connect(_ url: URL) {
        queue.async {
            self.url = url
            self.task?.cancel(with: .goingAway, reason: nil)
            let t = self.session.webSocketTask(with: url)
            t.maximumMessageSize = 8 * 1024 * 1024
            self.task = t
            self.sending = false
            self.state = .connecting
            t.resume()
            self.listen(t)
        }
    }

    func disconnect() {
        queue.async {
            self.url = nil
            self.task?.cancel(with: .normalClosure, reason: nil)
            self.task = nil
            self.state = .idle
        }
    }

    /// Returns false if the frame was dropped (not connected, or previous frame still sending).
    @discardableResult
    func send(_ data: Data) -> Bool {
        queue.sync {
            guard state == .connected, !sending, let t = task else { return false }
            sending = true
            t.send(.data(data)) { [weak self] error in
                guard let self else { return }
                self.queue.async {
                    self.sending = false
                    if let error { self.fail(error) }
                }
            }
            return true
        }
    }

    private func listen(_ t: URLSessionWebSocketTask) {
        // The Mac sends nothing; receiving just surfaces disconnects.
        t.receive { [weak self] result in
            guard let self else { return }
            if case .failure(let e) = result {
                self.queue.async { if self.task === t { self.fail(e) } }
            } else {
                self.listen(t)
            }
        }
    }

    private func fail(_ error: Error) {
        lastError = error.localizedDescription
        state = .failed
        task?.cancel(with: .abnormalClosure, reason: nil)
        task = nil
        // auto-retry while the user still wants to stream
        if let url {
            queue.asyncAfter(deadline: .now() + 2) { [weak self] in
                guard let self, self.url == url, self.state == .failed else { return }
                self.connect(url)
            }
        }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol protocol: String?) {
        queue.async { if self.task === webSocketTask { self.lastError = nil; self.state = .connected } }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        queue.async {
            if self.task === webSocketTask { self.fail(NSError(domain: "EchoSense", code: closeCode.rawValue,
                userInfo: [NSLocalizedDescriptionKey: "Mac closed the connection (\(closeCode.rawValue))"])) }
        }
    }
}
