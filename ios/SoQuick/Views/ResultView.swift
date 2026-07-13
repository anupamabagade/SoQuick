import SwiftUI
import Photos

struct ResultView: View {
    let frames: [FreezeFrame]
    @State private var savedIndices: Set<Int> = []
    @State private var saveError: String?

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
                        .padding(.top, 60)
                        .padding(.bottom, 32)
                    }

                    // ── Frame cards ────────────────────────────────────
                    VStack(spacing: 20) {
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
                                        // Badge
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

                                    // Image
                                    Image(uiImage: frame.image)
                                        .resizable()
                                        .scaledToFit()
                                        .frame(maxWidth: .infinity)

                                    // Stride + save row
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

                        // Save all button
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
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 20)
                    .padding(.bottom, 40)
                }
            }
        }
        .navigationBarHidden(true)
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
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
