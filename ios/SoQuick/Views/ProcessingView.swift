import SwiftUI

struct ProcessingView: View {
    @State private var pulse = false
    @State private var rotation = 0.0

    var body: some View {
        ZStack {
            Color.black.opacity(0.45).ignoresSafeArea()

            VStack(spacing: 24) {
                ZStack {
                    Circle()
                        .fill(Color.sqPastel.opacity(0.3))
                        .frame(width: 100, height: 100)
                        .scaleEffect(pulse ? 1.25 : 1.0)
                        .animation(.easeInOut(duration: 1.1).repeatForever(autoreverses: true),
                                   value: pulse)

                    SoftballIcon(size: 64)
                        .rotationEffect(.degrees(rotation))
                        .animation(.linear(duration: 4).repeatForever(autoreverses: false),
                                   value: rotation)
                }

                VStack(spacing: 8) {
                    Text("Analyzing Pitch…")
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                    Text("This takes 30–60 seconds")
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.7))
                }
            }
            .padding(36)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24))
        }
        .onAppear {
            pulse    = true
            rotation = 360
        }
    }
}
