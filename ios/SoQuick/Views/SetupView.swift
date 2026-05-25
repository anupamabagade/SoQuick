import SwiftUI
import PhotosUI

struct SetupView: View {
    @State private var viewType = "lateral"
    @State private var pHeight = 62
    @State private var pSide = "Right"
    @State private var displayMode = "All"
    @State private var slowMo = 2

    @State private var selectedItem: PhotosPickerItem?
    @State private var videoURL: URL?
    @State private var isProcessing = false
    @State private var resultURL: URL?
    @State private var errorMessage: String?

    private let displayModes = ["All", "Wrist Trace & Velocity Only", "Arm Angles Only", "Leg Angles Only"]

    var body: some View {
        NavigationStack {
            Form {
                Section("Analysis Mode") {
                    Picker("View", selection: $viewType) {
                        Text("Lateral (Trace)").tag("lateral")
                        Text("Back (Separation)").tag("back")
                    }
                    .pickerStyle(.segmented)
                }

                if viewType == "lateral" {
                    Section("Lateral Parameters") {
                        Stepper("Height: \(pHeight) in", value: $pHeight, in: 48...84)
                        Picker("Pitching Arm", selection: $pSide) {
                            Text("Right").tag("Right")
                            Text("Left").tag("Left")
                        }
                        Picker("Measurements", selection: $displayMode) {
                            ForEach(displayModes, id: \.self) { Text($0) }
                        }
                    }
                }

                Section("Slow Motion") {
                    Stepper("\(slowMo)×", value: $slowMo, in: 1...4)
                }

                Section("Video") {
                    PhotosPicker(
                        selection: $selectedItem,
                        matching: .videos,
                        photoLibrary: .shared()
                    ) {
                        Label(
                            videoURL == nil ? "Pick a Video" : "Video Selected ✓",
                            systemImage: "video.badge.plus"
                        )
                    }
                    .onChange(of: selectedItem) { _, item in
                        Task { await loadVideo(from: item) }
                    }
                }

                Section {
                    Button(action: runAnalysis) {
                        Label("Run Analysis", systemImage: "bolt.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(videoURL == nil || isProcessing)
                }
            }
            .navigationTitle("SoQuick")
            .navigationDestination(item: $resultURL) { url in
                ResultView(resultURL: url)
            }
            .overlay {
                if isProcessing {
                    ProcessingView()
                }
            }
            .alert("Error", isPresented: .constant(errorMessage != nil), actions: {
                Button("OK") { errorMessage = nil }
            }, message: {
                Text(errorMessage ?? "")
            })
        }
    }

    private func loadVideo(from item: PhotosPickerItem?) async {
        guard let item else { return }
        guard let movie = try? await item.loadTransferable(type: VideoTransferable.self) else { return }
        videoURL = movie.url
    }

    private func runAnalysis() {
        guard let videoURL else { return }
        isProcessing = true
        let params = AnalysisParams(
            viewType: viewType,
            pHeight: pHeight,
            pSide: pSide,
            displayMode: displayMode,
            slowMo: slowMo
        )
        Task {
            do {
                let url = try await AnalysisService.analyze(videoURL: videoURL, params: params)
                await MainActor.run {
                    isProcessing = false
                    resultURL = url
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

// Helper: lets PhotosPicker export a video to a temp file URL
struct VideoTransferable: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { video in
            SentTransferredFile(video.url)
        } importing: { received in
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("soquick_input_\(UUID().uuidString).mp4")
            try FileManager.default.copyItem(at: received.file, to: dest)
            return VideoTransferable(url: dest)
        }
    }
}
