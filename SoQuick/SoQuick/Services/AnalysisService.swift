import Foundation
import UIKit

struct FreezeFrame: Identifiable {
    let id = UUID()
    let label: String
    let image: UIImage
    let stride: [String: Double]
}

struct AnalysisResult {
    let frames: [FreezeFrame]
    let videoURL: URL?
}

enum AnalysisError: LocalizedError {
    case serverError(String)
    case noFrames
    case badResponse

    var errorDescription: String? {
        switch self {
        case .serverError(let msg): return "Server error: \(msg)"
        case .noFrames:            return "No key moments detected in this video."
        case .badResponse:         return "Unexpected response from server."
        }
    }
}

class AnalysisService {
    static let shared = AnalysisService()
    private let baseURL = "https://soquick.onrender.com"

    func analyze(videoURL: URL, height: Int, side: String) async throws -> AnalysisResult {
        let endpoint = URL(string: "\(baseURL)/analyze")!
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 300

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)",
                         forHTTPHeaderField: "Content-Type")

        let videoData = try Data(contentsOf: videoURL)
        var body = Data()
        body.appendFilePart(name: "video", filename: "pitch.mp4",
                            mimeType: "video/mp4", data: videoData, boundary: boundary)
        body.appendTextPart(name: "p_height", value: "\(height)", boundary: boundary)
        body.appendTextPart(name: "p_side",   value: side,        boundary: boundary)
        body.append("--\(boundary)--\r\n")

        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let http = response as? HTTPURLResponse else { throw AnalysisError.badResponse }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw AnalysisError.badResponse
        }

        if http.statusCode != 200 {
            let msg = json["error"] as? String ?? "HTTP \(http.statusCode)"
            throw AnalysisError.serverError(msg)
        }

        guard let framesJSON = json["freeze_frames"] as? [[String: Any]] else {
            throw AnalysisError.noFrames
        }

        let frames: [FreezeFrame] = framesJSON.compactMap { f in
            guard let label   = f["label"]     as? String,
                  let b64     = f["image_b64"] as? String,
                  let imgData = Data(base64Encoded: b64),
                  let image   = UIImage(data: imgData) else { return nil }
            let stride = f["stride"] as? [String: Double] ?? [:]
            return FreezeFrame(label: label, image: image, stride: stride)
        }

        if frames.isEmpty { throw AnalysisError.noFrames }

        // Decode arm-path video if present
        var armPathURL: URL? = nil
        if let videoB64 = json["video_b64"] as? String,
           !videoB64.isEmpty,
           let videoData = Data(base64Encoded: videoB64) {
            let tmp = FileManager.default.temporaryDirectory
                .appendingPathComponent("soquick_armpath_\(UUID().uuidString).mp4")
            try? videoData.write(to: tmp)
            armPathURL = tmp
        }

        return AnalysisResult(frames: frames, videoURL: armPathURL)
    }
}

private extension Data {
    mutating func appendFilePart(name: String, filename: String,
                                 mimeType: String, data: Data, boundary: String) {
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n")
        append("Content-Type: \(mimeType)\r\n\r\n")
        append(data)
        append("\r\n")
    }

    mutating func appendTextPart(name: String, value: String, boundary: String) {
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
        append(value)
        append("\r\n")
    }

    mutating func append(_ string: String) {
        if let d = string.data(using: .utf8) { append(d) }
    }
}
