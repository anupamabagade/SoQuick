import SwiftUI

struct PlayerDetailsView: View {
    @EnvironmentObject var auth: AuthService
    @Environment(\.dismiss) private var dismiss

    var isEditing: Bool = false

    @State private var name         = ""
    @State private var age          = 14
    @State private var heightFeet   = 5
    @State private var heightInches = 4
    @State private var weight       = ""
    @State private var handedness   = "right"
    @State private var level        = LevelOfPlay.travel14U
    @State private var fastestPitch = ""
    @State private var zipCode      = ""

    @State private var isLoading = false
    @State private var errorMsg: String?

    var body: some View {
        ZStack(alignment: .top) {
            Color.sqPastel.ignoresSafeArea()

            ScrollView {
                VStack(spacing: 0) {

                    // Header
                    ZStack {
                        sqGradient
                        HStack(alignment: .top, spacing: 0) {
                            // invisible balance matching the single button on the right
                            Image(systemName: "xmark")
                                .font(.system(size: 17, weight: .medium))
                                .padding(12)
                                .opacity(0)
                            Spacer()
                            VStack(spacing: 6) {
                                SoQuickBrand()
                                Text(isEditing ? "Edit Profile" : "Tell us about yourself")
                                    .font(.subheadline)
                                    .foregroundStyle(.white.opacity(0.85))
                            }
                            Spacer()
                            if isEditing {
                                Button { dismiss() } label: {
                                    Image(systemName: "xmark")
                                        .font(.system(size: 17, weight: .medium))
                                        .foregroundStyle(.white.opacity(0.85))
                                        .padding(12)
                                }
                            } else {
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

                    VStack(spacing: 16) {

                        // Name + Age
                        SQCard {
                            VStack(spacing: 0) {
                                inputRow("Full Name", icon: "person", text: $name)
                                Divider().padding(.horizontal, 16)
                                stepperRow("Age", value: $age, range: 8...30, unit: "yrs")
                            }
                        }

                        // Height + Weight
                        SQCard {
                            VStack(spacing: 0) {
                                heightRow
                                Divider().padding(.horizontal, 16)
                                numberRow("Weight", icon: "scalemass", text: $weight, unit: "lbs")
                            }
                        }

                        // Handedness
                        SQCard {
                            VStack(alignment: .leading, spacing: 10) {
                                Label("Throwing Hand", systemImage: "hand.raised")
                                    .font(.system(size: 15, weight: .medium))
                                    .foregroundStyle(Color.sqDark)
                                HStack(spacing: 10) {
                                    handButton("Right", value: "right")
                                    handButton("Left",  value: "left")
                                }
                            }
                            .padding(16)
                        }

                        // Level of play
                        SQCard {
                            VStack(alignment: .leading, spacing: 10) {
                                Label("Level of Play", systemImage: "trophy")
                                    .font(.system(size: 15, weight: .medium))
                                    .foregroundStyle(Color.sqDark)
                                Menu {
                                    ForEach(LevelOfPlay.allCases, id: \.self) { l in
                                        Button(l.display) { level = l }
                                    }
                                } label: {
                                    HStack {
                                        Text(level.display)
                                            .font(.system(size: 15, weight: .semibold))
                                            .foregroundStyle(Color.sqPrimary)
                                        Spacer()
                                        Image(systemName: "chevron.up.chevron.down")
                                            .font(.system(size: 13))
                                            .foregroundStyle(Color.sqMid)
                                    }
                                    .padding(12)
                                    .background(Color.sqPastel)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                                }
                            }
                            .padding(16)
                        }

                        // Optional fields
                        SQCard {
                            VStack(spacing: 0) {
                                numberRow("Fastest Pitch (optional)", icon: "speedometer",
                                         text: $fastestPitch, unit: "mph")
                                Divider().padding(.horizontal, 16)
                                inputRow("Zip Code", icon: "location", text: $zipCode, keyboard: .numberPad)
                            }
                        }

                        if let err = errorMsg {
                            HStack(spacing: 8) {
                                Image(systemName: "exclamationmark.circle.fill")
                                Text(err).font(.system(size: 14))
                            }
                            .foregroundStyle(.red)
                            .padding(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.red.opacity(0.08))
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                        }

                        Button(action: save) {
                            HStack(spacing: 10) {
                                if isLoading { ProgressView().tint(.white) }
                                else {
                                    Image(systemName: "checkmark.circle.fill")
                                    Text(isEditing ? "Save Changes" : "Save & Continue")
                                        .font(.system(size: 17, weight: .bold, design: .rounded))
                                }
                            }
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(sqGradient)
                            .clipShape(RoundedRectangle(cornerRadius: 16))
                            .shadow(color: Color.sqPrimary.opacity(0.3), radius: 10, x: 0, y: 5)
                        }
                        .disabled(isLoading)

                        if !isEditing {
                            Button {
                                Task { await auth.skipProfileSetup() }
                            } label: {
                                Text("Skip for now")
                                    .font(.system(size: 15))
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 20)
                    .padding(.bottom, 40)
                }
            }
        }
        .onAppear {
            guard isEditing, let p = auth.currentPlayer else { return }
            name         = p.name
            age          = p.age ?? 14
            heightFeet   = p.heightFeet ?? 5
            heightInches = p.heightInches ?? 4
            weight       = p.weightLbs.map { String(format: "%g", $0) } ?? ""
            handedness   = p.handedness ?? "right"
            level        = LevelOfPlay(rawValue: p.levelOfPlay ?? "") ?? .travel14U
            fastestPitch = p.fastestPitch.map { String(format: "%g", $0) } ?? ""
            zipCode      = p.zipCode ?? ""
        }
    }

    // MARK: - Sub-views

    @ViewBuilder
    private func inputRow(_ placeholder: String, icon: String,
                          text: Binding<String>,
                          keyboard: UIKeyboardType = .default) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon).foregroundStyle(Color.sqLight).frame(width: 20)
            TextField(placeholder, text: text)
                .keyboardType(keyboard)
                .autocorrectionDisabled()
        }
        .padding(16)
    }

    @ViewBuilder
    private func numberRow(_ placeholder: String, icon: String,
                           text: Binding<String>, unit: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon).foregroundStyle(Color.sqLight).frame(width: 20)
            TextField(placeholder, text: text).keyboardType(.decimalPad)
            Text(unit).font(.caption).foregroundStyle(.secondary)
        }
        .padding(16)
    }

    @ViewBuilder
    private func stepperRow(_ label: String, value: Binding<Int>,
                            range: ClosedRange<Int>, unit: String) -> some View {
        HStack {
            Label(label, systemImage: "calendar")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(Color.sqDark)
            Spacer()
            HStack(spacing: 0) {
                Button { if value.wrappedValue > range.lowerBound { value.wrappedValue -= 1 } } label: {
                    Image(systemName: "minus.circle.fill")
                        .font(.system(size: 22)).foregroundStyle(Color.sqLight)
                }
                Text("\(value.wrappedValue) \(unit)")
                    .font(.system(size: 16, weight: .bold, design: .rounded))
                    .foregroundStyle(Color.sqPrimary)
                    .frame(width: 64)
                Button { if value.wrappedValue < range.upperBound { value.wrappedValue += 1 } } label: {
                    Image(systemName: "plus.circle.fill")
                        .font(.system(size: 22)).foregroundStyle(Color.sqLight)
                }
            }
        }
        .padding(16)
    }

    @ViewBuilder
    private var heightRow: some View {
        HStack {
            Label("Height", systemImage: "ruler")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(Color.sqDark)
            Spacer()
            HStack(spacing: 4) {
                Picker("Feet", selection: $heightFeet) {
                    ForEach(4...7, id: \.self) { Text("\($0) ft") }
                }
                .pickerStyle(.menu)
                .tint(Color.sqPrimary)

                Picker("Inches", selection: $heightInches) {
                    ForEach(0...11, id: \.self) { Text("\($0) in") }
                }
                .pickerStyle(.menu)
                .tint(Color.sqPrimary)
            }
        }
        .padding(16)
    }

    @ViewBuilder
    private func handButton(_ label: String, value: String) -> some View {
        Button { handedness = value } label: {
            Text(label)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(handedness == value ? .white : Color.sqPrimary)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(handedness == value
                    ? AnyShapeStyle(sqGradient) : AnyShapeStyle(Color.sqPastel))
                .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    // MARK: - Save

    private func save() {
        errorMsg = nil
        guard !name.isEmpty else { errorMsg = "Please enter your name."; return }
        guard !zipCode.isEmpty else { errorMsg = "Please enter your zip code."; return }
        guard let wt = Double(weight), wt > 0 else {
            errorMsg = "Please enter a valid weight."
            return
        }

        isLoading = true
        Task {
            do {
                try await auth.savePlayerDetails(
                    name: name, age: age,
                    heightFeet: heightFeet, heightInches: heightInches,
                    weightLbs: wt, handedness: handedness,
                    levelOfPlay: level.rawValue,
                    fastestPitch: Double(fastestPitch),
                    zipCode: zipCode
                )
                if isEditing { dismiss() }
            } catch {
                errorMsg = error.localizedDescription
            }
            isLoading = false
        }
    }
}
