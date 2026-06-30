import SwiftUI
import PhotosUI

struct SetupView: View {
    @State private var pHeight = 65
    @State private var pSide   = "Right"

    @State private var selectedItem: PhotosPickerItem?
    @State private var videoURL: URL?

    @State private var isProcessing = false
    @State private var frames: [FreezeFrame]?
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            Form {
                Section("Video") {
                    PhotosPicker(
                        selection: $selectedItem,
                        matching: .videos,
                        photoLibrary: .shared()
                    ) {
                        Label(
                            videoURL == nil ? "Select Pitching Video" : "Video Selected ✓",
                            systemImage: "video.badge.plus"
                        )
                    }
                    .onChange(of: selectedItem) { _, item in
                        Task { await loadVideo(from: item) }
                    }
                }

                Section("Pitcher") {
                    Stepper("Height: \(pHeight) in", value: $pHeight, in: 48...84)
                    Picker("Pitching Arm", selection: $pSide) {
                        Text("Right").tag("Right")
                        Text("Left").tag("Left")
                    }
                    .pickerStyle(.segmented)
                }

                Section {
                    Button(action: runAnalysis) {
                        Label("Analyze Pitch", systemImage: "bolt.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(videoURL == nil || isProcessing)
                }
            }
            .navigationTitle("SoQuick")
            .navigationDestination(item: $frames) { f in
                ResultView(frames: f)
            }
            .overlay {
                if isProcessing { ProcessingView() }
            }
            .alert("Error", isPresented: .constant(errorMessage != nil)) {
                Button("OK") { errorMessage = nil }
            } message: {
                Text(errorMessage ?? "")
            }
        }
    }

    private func loadVideo(from item: PhotosPickerItem?) async {
        guard let item,
              let movie = try? await item.loadTransferable(type: VideoTransferable.self)
        else { return }
        videoURL = movie.url
    }

    private func runAnalysis() {
        guard let videoURL else { return }
        isProcessing = true
        Task {
            do {
                let result = try await AnalysisService.shared.analyze(
                    videoURL: videoURL, height: pHeight, side: pSide
                )
                await MainActor.run {
                    isProcessing = false
                    frames = result
                }
            } catch {
                await MainActor.run {
                    isProcessing = false
                    errorMessage = error.localizedDescription
                }
            }
        }
    }
}

// Exports the picked video to a temp file the app can read
struct VideoTransferable: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { video in
            SentTransferredFile(video.url)
        } importing: { received in
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("soquick_\(UUID().uuidString).mp4")
            try FileManager.default.copyItem(at: received.file, to: dest)
            return VideoTransferable(url: dest)
        }
    }
}

// Needed so [FreezeFrame] can drive navigationDestination
extension Array: @retroactive Identifiable where Element == FreezeFrame {
    public var id: Int { self.count }
}
