import SwiftUI
import AVKit
import Photos

struct ResultView: View {
    let resultURL: URL
    @State private var player: AVPlayer?
    @State private var savedToPhotos = false
    @State private var saveError: String?

    var body: some View {
        VStack(spacing: 0) {
            if let player {
                VideoPlayer(player: player)
                    .ignoresSafeArea(edges: .horizontal)
            }

            VStack(spacing: 16) {
                Button(action: saveToPhotos) {
                    Label(savedToPhotos ? "Saved!" : "Save to Photos", systemImage: savedToPhotos ? "checkmark.circle.fill" : "square.and.arrow.down")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(savedToPhotos ? .green : .blue)
                .disabled(savedToPhotos)
                .padding(.horizontal)

                if let saveError {
                    Text(saveError)
                        .foregroundStyle(.red)
                        .font(.caption)
                        .padding(.horizontal)
                }
            }
            .padding(.vertical, 20)
            .background(.ultraThinMaterial)
        }
        .navigationTitle("Analysis Result")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            player = AVPlayer(url: resultURL)
            player?.play()
        }
    }

    private func saveToPhotos() {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async { saveError = "Photo library access denied." }
                return
            }
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: resultURL)
            }) { success, error in
                DispatchQueue.main.async {
                    if success {
                        savedToPhotos = true
                    } else {
                        saveError = error?.localizedDescription ?? "Could not save video."
                    }
                }
            }
        }
    }
}
