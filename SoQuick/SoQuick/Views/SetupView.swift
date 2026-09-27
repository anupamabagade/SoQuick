import SwiftUI
import PhotosUI

struct SetupView: View {
    @EnvironmentObject var auth: AuthService
    @EnvironmentObject var store: AnalysisStore

    @State private var pHeight = 65
    @State private var pSide   = "Right"

    @State private var selectedItem: PhotosPickerItem?
    @State private var videoURL: URL?
    @State private var isLoadingVideo = false

    @State private var isProcessing = false
    @State private var frames: [FreezeFrame] = []
    @State private var armPathURL: URL?      = nil
    @State private var showResults   = false
    @State private var showDashboard = false
    @State private var showProfile   = false
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            ZStack(alignment: .top) {
                Color.sqPastel.ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 0) {
                        // ── Header ──────────────────────────────────────
                        ZStack {
                            sqGradient
                            HStack(alignment: .top, spacing: 0) {
                                VStack(spacing: 6) {
                                    SoQuickBrand()
                                    Text("Pitching Analysis")
                                        .font(.subheadline)
                                        .foregroundStyle(.white.opacity(0.85))
                                }
                                .frame(maxWidth: .infinity, alignment: .center)
                                HStack(spacing: 4) {
                                    Button { showProfile = true } label: {
                                        Image(systemName: "person.circle")
                                            .font(.system(size: 18, weight: .medium))
                                            .foregroundStyle(.white.opacity(0.85))
                                            .padding(12)
                                    }
                                    Button { showDashboard = true } label: {
                                        Image(systemName: "clock.arrow.trianglehead.counterclockwise.rotate.90")
                                            .font(.system(size: 18, weight: .medium))
                                            .foregroundStyle(.white.opacity(0.85))
                                            .padding(12)
                                    }
                                    Button {
                                        Task { try? await auth.signOut() }
                                    } label: {
                                        Image(systemName: "rectangle.portrait.and.arrow.right")
                                            .font(.system(size: 18, weight: .medium))
                                            .foregroundStyle(.white.opacity(0.85))
                                            .padding(12)
                                    }
                                }
                            }
                            .padding(.top, 48)
                            .padding(.bottom, 16)
                            .padding(.horizontal, 4)
                        }

                        // ── Cards ────────────────────────────────────────
                        VStack(spacing: 16) {
                            // Video picker card
                            SQCard {
                                PhotosPicker(
                                    selection: $selectedItem,
                                    matching: .videos,
                                    photoLibrary: .shared()
                                ) {
                                    HStack(spacing: 14) {
                                        ZStack {
                                            Circle()
                                                .fill(Color.sqPastel)
                                                .frame(width: 44, height: 44)
                                            if isLoadingVideo {
                                                ProgressView()
                                                    .tint(Color.sqPrimary)
                                            } else {
                                                Image(systemName: videoURL == nil
                                                      ? "video.badge.plus"
                                                      : "checkmark.circle.fill")
                                                    .font(.system(size: 20))
                                                    .foregroundStyle(videoURL == nil
                                                                     ? Color.sqPrimary : .green)
                                            }
                                        }
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text(isLoadingVideo
                                                 ? "Loading video…"
                                                 : videoURL == nil
                                                    ? "Select Pitching Video"
                                                    : "Video Selected")
                                                .font(.system(size: 16, weight: .semibold))
                                                .foregroundStyle(Color.sqDark)
                                            Text(isLoadingVideo
                                                 ? "Please wait"
                                                 : videoURL == nil
                                                    ? "Tap to choose from Photos"
                                                    : "Ready to analyze")
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        }
                                        Spacer()
                                        if !isLoadingVideo {
                                            Image(systemName: "chevron.right")
                                                .foregroundStyle(Color.sqMid)
                                        }
                                    }
                                    .padding(16)
                                }
                            }
                            .onChange(of: selectedItem) { _, item in
                                isLoadingVideo = true
                                videoURL = nil
                                Task {
                                    await loadVideo(from: item)
                                    isLoadingVideo = false
                                }
                            }

                            // Pitcher settings card
                            SQCard {
                                VStack(spacing: 0) {
                                    HStack {
                                        Label("Pitcher Height", systemImage: "ruler")
                                            .font(.system(size: 15, weight: .medium))
                                            .foregroundStyle(Color.sqDark)
                                        Spacer()
                                        HStack(spacing: 0) {
                                            Button { if pHeight > 48 { pHeight -= 1 } } label: {
                                                Image(systemName: "minus.circle.fill")
                                                    .font(.system(size: 22))
                                                    .foregroundStyle(Color.sqLight)
                                            }
                                            Text("\(pHeight) in")
                                                .font(.system(size: 16, weight: .bold, design: .rounded))
                                                .foregroundStyle(Color.sqPrimary)
                                                .frame(width: 58)
                                            Button { if pHeight < 84 { pHeight += 1 } } label: {
                                                Image(systemName: "plus.circle.fill")
                                                    .font(.system(size: 22))
                                                    .foregroundStyle(Color.sqLight)
                                            }
                                        }
                                    }
                                    .padding(16)

                                    Divider().padding(.horizontal, 16)

                                    VStack(alignment: .leading, spacing: 10) {
                                        Label("Pitching Arm", systemImage: "figure.softball")
                                            .font(.system(size: 15, weight: .medium))
                                            .foregroundStyle(Color.sqDark)
                                        HStack(spacing: 10) {
                                            ArmButton(label: "Right", selected: pSide == "Right") { pSide = "Right" }
                                            ArmButton(label: "Left",  selected: pSide == "Left")  { pSide = "Left"  }
                                        }
                                    }
                                    .padding(16)
                                }
                            }

                            // Analyze button
                            Button(action: runAnalysis) {
                                HStack(spacing: 10) {
                                    Image(systemName: "bolt.fill")
                                    Text("Analyze Pitch")
                                        .font(.system(size: 17, weight: .bold, design: .rounded))
                                }
                                .foregroundStyle(videoURL != nil ? .white : Color.gray)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 16)
                                .background(videoURL != nil
                                            ? AnyShapeStyle(sqGradient)
                                            : AnyShapeStyle(Color.gray.opacity(0.3)))
                                .clipShape(RoundedRectangle(cornerRadius: 16))
                                .shadow(color: Color.sqPrimary.opacity(videoURL != nil ? 0.35 : 0),
                                        radius: 10, x: 0, y: 5)
                            }
                            .disabled(videoURL == nil || isProcessing || isLoadingVideo)
                        }
                        .padding(.horizontal, 20)
                        .padding(.top, 20)
                        .padding(.bottom, 40)
                    }
                }
            }
            .toolbar(.hidden, for: .navigationBar)
            .navigationDestination(isPresented: $showResults) {
                ResultView(frames: frames, armPathURL: armPathURL)
            }
            .navigationDestination(isPresented: $showDashboard) {
                DashboardView()
            }
            .sheet(isPresented: $showProfile) {
                PlayerDetailsView(isEditing: true)
                    .environmentObject(auth)
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
                store.save(result: result)
                await MainActor.run {
                    isProcessing = false
                    frames       = result.frames
                    armPathURL   = result.videoURL
                    showResults  = true
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

private struct ArmButton: View {
    let label: String
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(selected ? .white : Color.sqPrimary)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(selected ? AnyShapeStyle(sqGradient) : AnyShapeStyle(Color.sqPastel))
                .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }
}

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
