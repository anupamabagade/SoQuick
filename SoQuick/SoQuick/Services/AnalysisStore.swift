import Foundation
import Combine
import UIKit

struct AnalysisRecord: Codable, Identifiable {
    let id: UUID
    let date: Date
    let labels: [String]
    let strides: [[String: Double]]
    let frameCount: Int
    var hasVideo: Bool = false
}

class AnalysisStore: ObservableObject {
    @Published var records: [AnalysisRecord] = []

    private let baseDir: URL
    private let indexFile: URL

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        baseDir  = docs.appendingPathComponent("SoQuickAnalyses", isDirectory: true)
        indexFile = baseDir.appendingPathComponent("index.json")
        try? FileManager.default.createDirectory(at: baseDir, withIntermediateDirectories: true)
        loadIndex()
    }

    func save(result: AnalysisResult) {
        let id  = UUID()
        let dir = baseDir.appendingPathComponent(id.uuidString)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        for (i, frame) in result.frames.enumerated() {
            if let data = frame.image.jpegData(compressionQuality: 0.85) {
                try? data.write(to: dir.appendingPathComponent("frame_\(i).jpg"))
            }
        }

        var hasVideo = false
        if let src = result.videoURL,
           let data = try? Data(contentsOf: src) {
            try? data.write(to: dir.appendingPathComponent("arm_path.mp4"))
            hasVideo = true
        }

        let record = AnalysisRecord(
            id: id, date: Date(),
            labels: result.frames.map { $0.label },
            strides: result.frames.map { $0.stride },
            frameCount: result.frames.count,
            hasVideo: hasVideo
        )
        records.insert(record, at: 0)
        saveIndex()
    }

    func loadFrames(for record: AnalysisRecord) -> [FreezeFrame] {
        let dir = baseDir.appendingPathComponent(record.id.uuidString)
        return (0..<record.frameCount).compactMap { i in
            guard let data  = try? Data(contentsOf: dir.appendingPathComponent("frame_\(i).jpg")),
                  let image = UIImage(data: data) else { return nil }
            return FreezeFrame(
                label:  record.labels[safe: i]  ?? "Frame \(i + 1)",
                image:  image,
                stride: record.strides[safe: i] ?? [:]
            )
        }
    }

    func videoURL(for record: AnalysisRecord) -> URL? {
        guard record.hasVideo else { return nil }
        let url = baseDir
            .appendingPathComponent(record.id.uuidString)
            .appendingPathComponent("arm_path.mp4")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    func loadResult(for record: AnalysisRecord) -> AnalysisResult {
        AnalysisResult(frames: loadFrames(for: record), videoURL: videoURL(for: record))
    }

    func delete(_ record: AnalysisRecord) {
        try? FileManager.default.removeItem(at: baseDir.appendingPathComponent(record.id.uuidString))
        records.removeAll { $0.id == record.id }
        saveIndex()
    }

    private func loadIndex() {
        guard let data   = try? Data(contentsOf: indexFile),
              let loaded = try? JSONDecoder().decode([AnalysisRecord].self, from: data)
        else { return }
        records = loaded
    }

    private func saveIndex() {
        if let data = try? JSONEncoder().encode(records) {
            try? data.write(to: indexFile)
        }
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
