import SwiftUI
import Photos

struct ResultView: View {
    let frames: [FreezeFrame]
    @State private var savedIndex: Int?
    @State private var saveError: String?

    var body: some View {
        ScrollView {
            VStack(spacing: 28) {
                ForEach(Array(frames.enumerated()), id: \.offset) { index, frame in
                    VStack(alignment: .leading, spacing: 8) {
                        Text(frame.label)
                            .font(.headline)
                            .padding(.horizontal)

                        Image(uiImage: frame.image)
                            .resizable()
                            .scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                            .padding(.horizontal)

                        if let horiz = frame.stride["horiz_ft"] {
                            Text(String(format: "Stride: %.1f ft", horiz))
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                                .padding(.horizontal)
                        }

                        Button {
                            saveImage(frame.image, index: index)
                        } label: {
                            Label(
                                savedIndex == index ? "Saved!" : "Save Image",
                                systemImage: savedIndex == index ? "checkmark.circle.fill" : "square.and.arrow.down"
                            )
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .tint(savedIndex == index ? .green : .blue)
                        .padding(.horizontal)
                    }
                }

                if let saveError {
                    Text(saveError)
                        .foregroundStyle(.red)
                        .font(.caption)
                        .padding()
                }
            }
            .padding(.vertical)
        }
        .navigationTitle("Key Moments")
        .navigationBarTitleDisplayMode(.inline)
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
                    if success {
                        savedIndex = index
                    } else {
                        saveError = error?.localizedDescription ?? "Could not save image."
                    }
                }
            }
        }
    }
}
