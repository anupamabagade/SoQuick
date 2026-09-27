import SwiftUI

struct DashboardView: View {
    @EnvironmentObject var store: AnalysisStore
    @Environment(\.dismiss) var dismiss

    @State private var selectedResult: AnalysisResult? = nil
    @State private var showResult = false

    private let columns = [GridItem(.flexible(), spacing: 12), GridItem(.flexible(), spacing: 12)]

    var body: some View {
        ZStack(alignment: .top) {
            Color.sqPastel.ignoresSafeArea()

            VStack(spacing: 0) {
                // Header
                ZStack {
                    sqGradient
                    HStack {
                        Button { dismiss() } label: {
                            Image(systemName: "chevron.left")
                                .font(.system(size: 18, weight: .semibold))
                                .foregroundStyle(.white)
                                .padding(8)
                        }
                        Spacer()
                        VStack(spacing: 4) {
                            SoQuickBrand()
                            Text("Analysis History")
                                .font(.subheadline)
                                .foregroundStyle(.white.opacity(0.85))
                        }
                        Spacer()
                        // Balance the back button
                        Image(systemName: "chevron.left").opacity(0).padding(8)
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 48)
                    .padding(.bottom, 16)
                }
                .fixedSize(horizontal: false, vertical: true)

                if store.records.isEmpty {
                    Spacer()
                    VStack(spacing: 16) {
                        Image(systemName: "film.stack")
                            .font(.system(size: 52))
                            .foregroundStyle(Color.sqMid)
                        Text("No analyses yet")
                            .font(.system(size: 20, weight: .bold, design: .rounded))
                            .foregroundStyle(Color.sqDark)
                        Text("Analyse a pitch and it will appear here.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                } else {
                    ScrollView {
                        LazyVGrid(columns: columns, spacing: 12) {
                            ForEach(store.records) { record in
                                AnalysisCard(record: record)
                                    .onTapGesture {
                                        selectedResult = store.loadResult(for: record)
                                        showResult = true
                                    }
                                    .contextMenu {
                                        Button(role: .destructive) {
                                            store.delete(record)
                                        } label: {
                                            Label("Delete", systemImage: "trash")
                                        }
                                    }
                            }
                        }
                        .padding(16)
                    }
                }
            }
        }
        .navigationDestination(isPresented: $showResult) {
            if let r = selectedResult {
                ResultView(frames: r.frames, armPathURL: r.videoURL)
            }
        }
    }
}

// MARK: - Card

private struct AnalysisCard: View {
    let record: AnalysisRecord
    @EnvironmentObject var store: AnalysisStore
    @State private var thumbnails: [UIImage] = []

    private let thumbColumns = [GridItem(.flexible(), spacing: 2), GridItem(.flexible(), spacing: 2)]

    var body: some View {
        SQCard {
            VStack(alignment: .leading, spacing: 0) {
                // 2×2 thumbnail grid
                LazyVGrid(columns: thumbColumns, spacing: 2) {
                    ForEach(0..<4, id: \.self) { i in
                        Group {
                            if i < thumbnails.count {
                                Image(uiImage: thumbnails[i])
                                    .resizable()
                                    .scaledToFill()
                            } else {
                                Color.sqPastel
                            }
                        }
                        .frame(height: 65)
                        .clipped()
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .padding([.top, .horizontal], 8)

                VStack(alignment: .leading, spacing: 2) {
                    Text(record.date.formatted(date: .abbreviated, time: .omitted))
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(Color.sqDark)
                    Text(record.date.formatted(date: .omitted, time: .shortened))
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                    Text("\(record.frameCount) key moments")
                        .font(.system(size: 11))
                        .foregroundStyle(Color.sqLight)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
            }
        }
        .onAppear {
            thumbnails = store.loadFrames(for: record).map { $0.image }
        }

    }
}
