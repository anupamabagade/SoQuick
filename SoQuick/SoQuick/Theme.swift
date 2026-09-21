import SwiftUI

// MARK: - Brand colors
extension Color {
    static let sqPrimary = Color(red: 0.49, green: 0.23, blue: 0.93)  // #7C3AED deep purple
    static let sqLight   = Color(red: 0.65, green: 0.55, blue: 0.98)  // #A78BFA medium lavender
    static let sqPastel  = Color(red: 0.93, green: 0.91, blue: 1.00)  // #EDE9FE soft lavender bg
    static let sqMid     = Color(red: 0.77, green: 0.71, blue: 0.99)  // #C4B5FD light accent
    static let sqDark    = Color(red: 0.12, green: 0.11, blue: 0.29)  // #1E1B4B deep text
}

var sqGradient: LinearGradient {
    LinearGradient(
        colors: [Color.sqLight, Color.sqPrimary],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
}

// MARK: - Softball icon
struct SoftballIcon: View {
    var size: CGFloat = 40

    var body: some View {
        ZStack {
            Circle()
                .fill(.white)
                .frame(width: size, height: size)
                .shadow(color: .black.opacity(0.12), radius: 6, x: 0, y: 3)
            // Left seam arc
            Arc(startDegrees: -50, endDegrees: 50, clockwise: false)
                .stroke(Color.sqPrimary,
                        style: StrokeStyle(lineWidth: size * 0.08, lineCap: .round))
                .frame(width: size * 0.45, height: size * 0.45)
                .offset(x: -size * 0.12)
            // Right seam arc
            Arc(startDegrees: 130, endDegrees: 230, clockwise: false)
                .stroke(Color.sqPrimary,
                        style: StrokeStyle(lineWidth: size * 0.08, lineCap: .round))
                .frame(width: size * 0.45, height: size * 0.45)
                .offset(x: size * 0.12)
        }
        .frame(width: size, height: size)
    }
}

private struct Arc: Shape {
    var startDegrees: Double
    var endDegrees: Double
    var clockwise: Bool

    func path(in rect: CGRect) -> Path {
        var p = Path()
        p.addArc(center: CGPoint(x: rect.midX, y: rect.midY),
                 radius: min(rect.width, rect.height) / 2,
                 startAngle: .degrees(startDegrees),
                 endAngle: .degrees(endDegrees),
                 clockwise: clockwise)
        return p
    }
}

// MARK: - Shared header used on every screen
struct SoQuickBrand: View {
    var body: some View {
        HStack(spacing: 12) {
            SoftballIcon(size: 44)
            Text("SoQuick")
                .font(.system(size: 30, weight: .bold, design: .rounded))
                .foregroundStyle(.white)
        }
    }
}

// MARK: - Gradient card background
struct SQCard<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        content
            .background(Color.white)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .shadow(color: Color.sqPrimary.opacity(0.12), radius: 12, x: 0, y: 4)
    }
}
