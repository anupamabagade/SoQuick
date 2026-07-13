import SwiftUI

struct ProcessingView: View {
    var body: some View {
        ZStack {
            Color.black.opacity(0.55).ignoresSafeArea()
            VStack(spacing: 20) {
                ProgressView()
                    .scaleEffect(1.6)
                    .tint(.white)
                Text("Analyzing pitch…")
                    .foregroundStyle(.white)
                    .font(.headline)
                Text("This takes 30–90 seconds")
                    .foregroundStyle(.white.opacity(0.7))
                    .font(.caption)
            }
            .padding(32)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        }
    }
}
