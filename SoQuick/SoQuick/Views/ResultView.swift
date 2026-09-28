import SwiftUI
import Photos
import AVKit

struct ResultView: View {
    let frames: [FreezeFrame]
    let armPathURL: URL?
    @EnvironmentObject var store: AnalysisStore
    @Environment(\.dismiss) var dismiss
    @State private var savedIndices: Set<Int> = []
    @State private var savedVideo = false
    @State private var saveError: String?
    @State private var showDashboard = false
    @State private var player: AVPlayer? = nil
    @State private var showEnlargedVideo = false
    @State private var enlargedImage: EnlargedImage? = nil

    private let momentEmojis = ["🦵", "🏔️", "👣", "⚾"]

    var body: some View {
        ZStack(alignment: .top) {
            Color.sqPastel.ignoresSafeArea()

            ScrollView {
                VStack(spacing: 0) {
                    // ── Header ─────────────────────────────────────────
                    ZStack {
                        sqGradient.ignoresSafeArea(edges: .top)
                        VStack(spacing: 6) {
                            SoQuickBrand()
                            Text("\(frames.count) Key Moments Detected")
                                .font(.subheadline)
                                .foregroundStyle(.white.opacity(0.85))
                        }
                        .padding(.top, 48)
                        .padding(.bottom, 16)
                        .frame(maxWidth: .infinity)

                        HStack {
                            Button { dismiss() } label: {
                                Image(systemName: "chevron.left")
                                    .font(.system(size: 18, weight: .semibold))
                                    .foregroundStyle(.white)
                                    .padding(12)
                            }
                            Spacer()
                            Button { showDashboard = true } label: {
                                Image(systemName: "clock.arrow.trianglehead.counterclockwise.rotate.90")
                                    .font(.system(size: 18, weight: .medium))
                                    .foregroundStyle(.white)
                                    .padding(12)
                            }
                        }
                        .padding(.horizontal, 8)
                        .padding(.top, 52)
                    }

                    VStack(spacing: 20) {

                        // ── Arm Path Video ─────────────────────────────
                        if let p = player {
                            SQCard {
                                VStack(alignment: .leading, spacing: 0) {
                                    HStack(spacing: 10) {
                                        Text("🎥")
                                            .font(.title2)
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text("Arm Path")
                                                .font(.system(size: 16, weight: .bold))
                                                .foregroundStyle(Color.sqDark)
                                            Text("Wrist heatmap trace")
                                                .font(.caption)
                                                .foregroundStyle(Color.sqLight)
                                        }
                                        Spacer()
                                    }
                                    .padding(.horizontal, 16)
                                    .padding(.top, 16)
                                    .padding(.bottom, 12)

                                    ZStack(alignment: .topTrailing) {
                                        VideoPlayer(player: p)
                                            .frame(height: 220)
                                            .onAppear { p.play() }
                                            .contentShape(Rectangle())
                                            .onTapGesture { showEnlargedVideo = true }

                                        Image(systemName: "arrow.up.left.and.arrow.down.right")
                                            .font(.system(size: 13, weight: .semibold))
                                            .foregroundStyle(.white)
                                            .padding(8)
                                            .background(.black.opacity(0.45))
                                            .clipShape(Circle())
                                            .padding(10)
                                            .allowsHitTesting(false)
                                    }

                                    HStack {
                                        Spacer()
                                        Button {
                                            if let url = armPathURL { saveVideo(url) }
                                        } label: {
                                            Label(
                                                savedVideo ? "Saved" : "Save Video",
                                                systemImage: savedVideo
                                                    ? "checkmark.circle.fill"
                                                    : "square.and.arrow.down"
                                            )
                                            .font(.system(size: 13, weight: .semibold))
                                            .foregroundStyle(savedVideo ? .green : Color.sqPrimary)
                                            .padding(.horizontal, 12)
                                            .padding(.vertical, 6)
                                            .background(
                                                savedVideo ? Color.green.opacity(0.12) : Color.sqPastel
                                            )
                                            .clipShape(Capsule())
                                        }
                                    }
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 12)
                                }
                            }
                        }

                        // ── Frame cards ────────────────────────────────
                        ForEach(Array(frames.enumerated()), id: \.offset) { index, frame in
                            SQCard {
                                VStack(alignment: .leading, spacing: 0) {
                                    // Label row
                                    HStack(spacing: 10) {
                                        Text(momentEmojis[safe: index] ?? "📸")
                                            .font(.title2)
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text(frame.label)
                                                .font(.system(size: 16, weight: .bold))
                                                .foregroundStyle(Color.sqDark)
                                            Text("Frame \(index + 1) of \(frames.count)")
                                                .font(.caption)
                                                .foregroundStyle(Color.sqLight)
                                        }
                                        Spacer()
                                        Text("\(index + 1)")
                                            .font(.system(size: 13, weight: .bold))
                                            .foregroundStyle(.white)
                                            .frame(width: 28, height: 28)
                                            .background(sqGradient)
                                            .clipShape(Circle())
                                    }
                                    .padding(.horizontal, 16)
                                    .padding(.top, 16)
                                    .padding(.bottom, 12)

                                    ZStack(alignment: .topTrailing) {
                                        Image(uiImage: frame.image)
                                            .resizable()
                                            .scaledToFit()
                                            .frame(maxWidth: .infinity)
                                            .contentShape(Rectangle())
                                            .onTapGesture { enlargedImage = EnlargedImage(image: frame.image) }

                                        Image(systemName: "arrow.up.left.and.arrow.down.right")
                                            .font(.system(size: 13, weight: .semibold))
                                            .foregroundStyle(.white)
                                            .padding(8)
                                            .background(.black.opacity(0.45))
                                            .clipShape(Circle())
                                            .padding(10)
                                            .allowsHitTesting(false)
                                    }

                                    HStack {
                                        if let horiz = frame.stride["horiz_ft"] {
                                            Label(
                                                String(format: "Stride: %.1f ft", horiz),
                                                systemImage: "arrow.left.and.right"
                                            )
                                            .font(.system(size: 13, weight: .medium))
                                            .foregroundStyle(Color.sqPrimary)
                                            .padding(.horizontal, 10)
                                            .padding(.vertical, 5)
                                            .background(Color.sqPastel)
                                            .clipShape(Capsule())
                                        }
                                        Spacer()
                                        Button {
                                            saveImage(frame.image, index: index)
                                        } label: {
                                            Label(
                                                savedIndices.contains(index) ? "Saved" : "Save",
                                                systemImage: savedIndices.contains(index)
                                                    ? "checkmark.circle.fill"
                                                    : "square.and.arrow.down"
                                            )
                                            .font(.system(size: 13, weight: .semibold))
                                            .foregroundStyle(
                                                savedIndices.contains(index) ? .green : Color.sqPrimary
                                            )
                                            .padding(.horizontal, 12)
                                            .padding(.vertical, 6)
                                            .background(
                                                savedIndices.contains(index)
                                                    ? Color.green.opacity(0.12)
                                                    : Color.sqPastel
                                            )
                                            .clipShape(Capsule())
                                        }
                                    }
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 12)
                                }
                            }
                        }

                        if let err = saveError {
                            Text(err)
                                .font(.caption)
                                .foregroundStyle(.red)
                                .padding()
                        }

                        if !frames.isEmpty {
                            Button {
                                frames.enumerated().forEach { i, f in saveImage(f.image, index: i) }
                            } label: {
                                HStack(spacing: 8) {
                                    Image(systemName: "square.and.arrow.down.on.square")
                                    Text("Save All Images")
                                        .font(.system(size: 16, weight: .bold, design: .rounded))
                                }
                                .foregroundStyle(.white)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 16)
                                .background(sqGradient)
                                .clipShape(RoundedRectangle(cornerRadius: 16))
                                .shadow(color: Color.sqPrimary.opacity(0.3), radius: 8, x: 0, y: 4)
                            }
                        }

                        HStack(spacing: 12) {
                            Button { dismiss() } label: {
                                HStack(spacing: 8) {
                                    Image(systemName: "video.badge.plus")
                                    Text("New Analysis")
                                        .font(.system(size: 15, weight: .semibold, design: .rounded))
                                }
                                .foregroundStyle(Color.sqPrimary)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 14)
                                .background(Color.sqPastel)
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                            }
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 20)
                    .padding(.bottom, 40)
                }
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .navigationDestination(isPresented: $showDashboard) {
            DashboardView()
        }
        .onAppear {
            if let url = armPathURL {
                player = AVPlayer(url: url)
            }
        }
        .onDisappear {
            player?.pause()
        }
        .fullScreenCover(isPresented: $showEnlargedVideo) {
            ZStack {
                Color.black.ignoresSafeArea()
                if let p = player {
                    VideoPlayer(player: p)
                        .ignoresSafeArea()
                }
                VStack {
                    HStack {
                        Spacer()
                        Button { showEnlargedVideo = false } label: {
                            Image(systemName: "xmark")
                                .font(.system(size: 18, weight: .semibold))
                                .foregroundStyle(.white)
                                .padding(12)
                                .background(.black.opacity(0.4))
                                .clipShape(Circle())
                        }
                        .padding()
                    }
                    Spacer()
                }
            }
        }
        .fullScreenCover(item: $enlargedImage) { item in
            ZoomableImageViewer(image: item.image) { enlargedImage = nil }
        }
    }

    private func saveImage(_ image: UIImage, index: Int) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async { saveError = "Photo library access denied." }
                return
            }
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAsset(from: image)
            }) { success, error in
                DispatchQueue.main.async {
                    if success { savedIndices.insert(index) }
                    else { saveError = error?.localizedDescription ?? "Could not save." }
                }
            }
        }
    }

    private func saveVideo(_ url: URL) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async { saveError = "Photo library access denied." }
                return
            }
            PHPhotoLibrary.shared().performChanges({
                PHAssetCreationRequest.forAsset().addResource(with: .video, fileURL: url, options: nil)
            }) { success, error in
                DispatchQueue.main.async {
                    if success { savedVideo = true }
                    else { saveError = error?.localizedDescription ?? "Could not save video." }
                }
            }
        }
    }
}

private struct EnlargedImage: Identifiable {
    let id = UUID()
    let image: UIImage
}

private struct ZoomableImageViewer: View {
    let image: UIImage
    let onDismiss: () -> Void

    @State private var scale: CGFloat = 1
    @State private var lastScale: CGFloat = 1
    @State private var offset: CGSize = .zero
    @State private var lastOffset: CGSize = .zero

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .scaleEffect(scale)
                .offset(offset)
                .gesture(
                    MagnificationGesture()
                        .onChanged { value in
                            scale = max(1, min(lastScale * value, 5))
                        }
                        .onEnded { _ in lastScale = scale }
                )
                .simultaneousGesture(
                    DragGesture()
                        .onChanged { value in
                            guard scale > 1 else { return }
                            offset = CGSize(width: lastOffset.width + value.translation.width,
                                            height: lastOffset.height + value.translation.height)
                        }
                        .onEnded { _ in lastOffset = offset }
                )
                .onTapGesture(count: 2) {
                    withAnimation {
                        scale = 1; lastScale = 1
                        offset = .zero; lastOffset = .zero
                    }
                }

            VStack {
                HStack {
                    Spacer()
                    Button(action: onDismiss) {
                        Image(systemName: "xmark")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundStyle(.white)
                            .padding(12)
                            .background(.black.opacity(0.4))
                            .clipShape(Circle())
                    }
                    .padding()
                }
                Spacer()
            }
        }
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
