import Foundation

struct AnalysisParams {
    var viewType: String       // "lateral" or "back"
    var pHeight: Int           // inches
    var pSide: String          // "Right" or "Left"
    var displayMode: String    // "All" / "Wrist Trace & Velocity Only" / "Arm Angles Only" / "Leg Angles Only"
    var slowMo: Int            // 1–4
}

enum AnalysisError: LocalizedError {
    case serverError(String)
    case noData

    var errorDescription: String? {
        switch self {
        case .serverError(let msg): return "Server error: \(msg)"
        case .noData: return "No data returned from server."
        }
    }
}

class AnalysisService {
    // Replace with your deployed Render URL after deployment.
    // For local testing use: http://localhost:8000
    static let baseURL = "http://localhost:8000"

    static func analyze(videoURL: URL, params: AnalysisParams) async throws -> URL {
        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: URL(string: "\(baseURL)/analyze")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 300
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()

        func appendField(_ name: String, _ value: String) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(value)\r\n".data(using: .utf8)!)
        }

        appendField("view_type",    params.viewType)
        appendField("p_height",     "\(params.pHeight)")
        appendField("p_side",       params.pSide)
        appendField("display_mode", params.displayMode)
        appendField("slow_mo",      "\(params.slowMo)")

        let videoData = try Data(contentsOf: videoURL)
        let filename = videoURL.lastPathComponent
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"video\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: video/mp4\r\n\r\n".data(using: .utf8)!)
        body.append(videoData)
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let http = response as? HTTPURLResponse else { throw AnalysisError.noData }
        guard http.statusCode == 200 else {
            let msg = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw AnalysisError.serverError("HTTP \(http.statusCode): \(msg)")
        }

        let resultURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("soquick_result_\(UUID().uuidString).mp4")
        try data.write(to: resultURL)
        return resultURL
    }
}
