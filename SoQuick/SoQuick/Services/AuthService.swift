import Foundation
import Combine
import Supabase

@MainActor
class AuthService: ObservableObject {

    let supabase = SupabaseClient(
        supabaseURL: URL(string: "https://okocqgfmybhbfveefhxl.supabase.co")!,
        supabaseKey: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9rb2NxZ2ZteWJoYmZ2ZWVmaHhsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODM5MDg5ODMsImV4cCI6MjA5OTQ4NDk4M30.TNJgbIxC0d8Xs_nFuB0tF6YVPEPkfqn8pAGKoPFxC8M"
    )

    @Published var isLoggedIn         = false
    @Published var isEmailPending     = false   // waiting for email confirmation
    @Published var needsPasswordReset = false
    @Published var profileComplete    = false
    @Published var isLoadingProfile   = false
    @Published var userRole: UserRole = .athlete
    @Published var userName           = ""
    @Published var inviteCode: String = ""
    @Published var rosterPlayers: [Player] = []
    @Published var currentPlayer: Player?

    private(set) var pendingEmail = ""

    init() {
        Task { await restoreSession() }
    }

    // MARK: - Session restore

    func restoreSession() async {
        guard let session = try? await supabase.auth.session else { return }
        isLoggedIn = true
        await loadProfile(userId: session.user.id)
    }

    // MARK: - Deep link handler

    func handleDeepLink(url: URL) async {
        do {
            try await supabase.auth.session(from: url)
            let type = extractType(from: url)
            if type == "recovery" {
                needsPasswordReset = true
            } else {
                isEmailPending = false
                isLoggedIn = true
                if let session = try? await supabase.auth.session {
                    await loadProfile(userId: session.user.id)
                }
            }
        } catch {
            print("Deep link error: \(error)")
        }
    }

    private func extractType(from url: URL) -> String? {
        let components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        if let type = components?.queryItems?.first(where: { $0.name == "type" })?.value {
            return type
        }
        if let fragment = components?.fragment {
            for part in fragment.components(separatedBy: "&") {
                let kv = part.components(separatedBy: "=")
                if kv.count == 2, kv[0] == "type" { return kv[1] }
            }
        }
        return nil
    }

    // MARK: - Auth actions

    func signIn(email: String, password: String) async throws {
        try await supabase.auth.signIn(email: email, password: password)
        let session = try await supabase.auth.session
        isLoggedIn = true
        await loadProfile(userId: session.user.id)
    }

    func signUp(email: String, password: String,
                fullName: String, role: UserRole) async throws {
        let response = try await supabase.auth.signUp(
            email: email,
            password: password,
            data: [
                "full_name": .string(fullName),
                "role": .string(role.rawValue)
            ]
        )
        if response.session != nil {
            isLoggedIn = true
            userRole   = role
            userName   = fullName
            profileComplete = (role == .coach)
        } else {
            // Email confirmation required
            isEmailPending = true
            pendingEmail   = email
        }
    }

    func resendConfirmationEmail() async throws {
        try await supabase.auth.resend(
            email: pendingEmail,
            type: .signup,
            emailRedirectTo: URL(string: "soquick://login-callback")
        )
    }

    func resetPassword(email: String) async throws {
        try await supabase.auth.resetPasswordForEmail(
            email,
            redirectTo: URL(string: "soquick://login-callback?type=recovery")
        )
    }

    func skipProfileSetup() async {
        guard let session = try? await supabase.auth.session else { return }
        try? await supabase.from("profiles")
            .update(["profile_complete": true])
            .eq("id", value: session.user.id)
            .execute()
        profileComplete = true
    }

    func updatePassword(_ newPassword: String) async throws {
        try await supabase.auth.update(user: UserAttributes(password: newPassword))
        needsPasswordReset = false
        isLoggedIn = true
        if let session = try? await supabase.auth.session {
            await loadProfile(userId: session.user.id)
        }
    }

    func signOut() async throws {
        try await supabase.auth.signOut()
        isLoggedIn         = false
        isEmailPending     = false
        needsPasswordReset = false
        profileComplete    = false
        userRole           = .athlete
        userName           = ""
        inviteCode         = ""
        rosterPlayers      = []
    }

    // MARK: - Player details (athletes)

    func loadPlayerDetails() async {
        guard let session = try? await supabase.auth.session else { return }
        currentPlayer = try? await supabase
            .from("players")
            .select()
            .eq("user_id", value: session.user.id)
            .single()
            .execute()
            .value
    }

    func savePlayerDetails(
        name: String, age: Int,
        heightFeet: Int, heightInches: Int,
        weightLbs: Double, handedness: String,
        levelOfPlay: String, fastestPitch: Double?,
        zipCode: String
    ) async throws {
        guard let session = try? await supabase.auth.session else { return }
        let userId = session.user.id

        struct PlayerInsert: Encodable {
            let userId: UUID
            let name: String
            let age: Int
            let heightFeet: Int
            let heightInches: Int
            let weightLbs: Double
            let handedness: String
            let levelOfPlay: String
            let fastestPitch: Double?
            let zipCode: String

            enum CodingKeys: String, CodingKey {
                case userId = "user_id"
                case name, age
                case heightFeet   = "height_feet"
                case heightInches = "height_inches"
                case weightLbs    = "weight_lbs"
                case handedness
                case levelOfPlay  = "level_of_play"
                case fastestPitch = "fastest_pitch"
                case zipCode      = "zip_code"
            }
        }

        // Mark profile complete first so sign-in routing works even if player upsert fails
        try await supabase.from("profiles")
            .update(["profile_complete": true])
            .eq("id", value: userId)
            .execute()

        try await supabase.from("players")
            .upsert(PlayerInsert(
                userId: userId, name: name, age: age,
                heightFeet: heightFeet, heightInches: heightInches,
                weightLbs: weightLbs, handedness: handedness,
                levelOfPlay: levelOfPlay, fastestPitch: fastestPitch,
                zipCode: zipCode
            ), onConflict: "user_id")
            .execute()

        profileComplete = true
        userName = name
        await loadPlayerDetails()
    }

    // MARK: - Coach roster

    func loadRosterPlayers() async {
        guard let session = try? await supabase.auth.session else { return }
        let userId = session.user.id
        do {
            // Manually added players
            let manual: [Player] = try await supabase
                .from("players")
                .select()
                .eq("added_by", value: userId)
                .execute()
                .value

            // Players who joined via invite code
            struct RosterEntry: Decodable { let playerId: UUID
                enum CodingKeys: String, CodingKey { case playerId = "player_id" }
            }
            let entries: [RosterEntry] = try await supabase
                .from("coach_roster")
                .select("player_id")
                .eq("coach_id", value: userId)
                .execute()
                .value

            var linked: [Player] = []
            for entry in entries {
                if let p = try? await supabase
                    .from("players").select()
                    .eq("id", value: entry.playerId)
                    .single().execute().value as Player {
                    linked.append(p)
                }
            }

            // Merge, deduplicate by id
            var seen = Set<UUID>()
            var all: [Player] = []
            for p in manual + linked {
                if seen.insert(p.id).inserted { all.append(p) }
            }
            rosterPlayers = all
        } catch {
            print("Roster load error: \(error)")
        }
    }

    func addManualPlayer(
        name: String, age: Int,
        heightFeet: Int, heightInches: Int,
        weightLbs: Double, handedness: String,
        levelOfPlay: String, fastestPitch: Double?,
        zipCode: String
    ) async throws {
        guard let session = try? await supabase.auth.session else { return }
        let coachId = session.user.id

        struct ManualInsert: Encodable {
            let addedBy: UUID
            let name: String; let age: Int
            let heightFeet: Int; let heightInches: Int
            let weightLbs: Double; let handedness: String
            let levelOfPlay: String; let fastestPitch: Double?
            let zipCode: String
            enum CodingKeys: String, CodingKey {
                case addedBy = "added_by"
                case name, age
                case heightFeet = "height_feet"; case heightInches = "height_inches"
                case weightLbs = "weight_lbs"; case handedness
                case levelOfPlay = "level_of_play"; case fastestPitch = "fastest_pitch"
                case zipCode = "zip_code"
            }
        }

        try await supabase.from("players")
            .insert(ManualInsert(
                addedBy: coachId, name: name, age: age,
                heightFeet: heightFeet, heightInches: heightInches,
                weightLbs: weightLbs, handedness: handedness,
                levelOfPlay: levelOfPlay, fastestPitch: fastestPitch,
                zipCode: zipCode
            ))
            .execute()

        await loadRosterPlayers()
    }

    func getOrCreateInviteCode() async throws {
        guard let session = try? await supabase.auth.session else { return }
        let userId = session.user.id

        struct CodeRow: Decodable { let inviteCode: String?
            enum CodingKeys: String, CodingKey { case inviteCode = "invite_code" }
        }
        let row: CodeRow = try await supabase
            .from("profiles").select("invite_code")
            .eq("id", value: userId).single().execute().value

        if let code = row.inviteCode {
            inviteCode = code
            return
        }

        // Generate a new one via DB function
        struct CodeResult: Decodable { let generateInviteCode: String
            enum CodingKeys: String, CodingKey { case generateInviteCode = "generate_invite_code" }
        }
        let result: CodeResult = try await supabase.rpc("generate_invite_code").execute().value
        let newCode = result.generateInviteCode

        try await supabase.from("profiles")
            .update(["invite_code": newCode])
            .eq("id", value: userId)
            .execute()

        inviteCode = newCode
    }

    func joinCoachByCode(_ code: String) async throws {
        guard let session = try? await supabase.auth.session else { return }
        let playerId = session.user.id

        // Find coach with this invite code
        struct CoachRow: Decodable { let id: UUID }
        let coach: CoachRow = try await supabase
            .from("profiles").select("id")
            .eq("invite_code", value: code)
            .eq("role", value: "coach")
            .single().execute().value

        // Find player's own player record
        struct PlayerRow: Decodable { let id: UUID }
        let playerRecord: PlayerRow = try await supabase
            .from("players").select("id")
            .eq("user_id", value: playerId)
            .single().execute().value

        // Insert roster entry
        struct RosterInsert: Encodable {
            let coachId: UUID; let playerId: UUID
            enum CodingKeys: String, CodingKey {
                case coachId = "coach_id"; case playerId = "player_id"
            }
        }
        try await supabase.from("coach_roster")
            .insert(RosterInsert(coachId: coach.id, playerId: playerRecord.id))
            .execute()
    }

    // MARK: - Profile

    private func loadProfile(userId: UUID) async {
        isLoadingProfile = true
        defer { isLoadingProfile = false }
        do {
            let profile: ProfileRow = try await supabase
                .from("profiles")
                .select("id, full_name, role, profile_complete, invite_code")
                .eq("id", value: userId).single().execute().value
            userRole        = UserRole(rawValue: profile.role ?? "athlete") ?? .athlete
            userName        = profile.fullName ?? ""
            inviteCode      = profile.inviteCode ?? ""

            if profile.profileComplete == true {
                profileComplete = true
                if userRole == .athlete { await loadPlayerDetails() }
            } else if userRole == .athlete {
                // Fallback: treat as complete if a player record already exists
                let resp = try await supabase
                    .from("players")
                    .select("id", count: .exact)
                    .eq("user_id", value: userId)
                    .execute()
                profileComplete = (resp.count ?? 0) > 0
            } else {
                profileComplete = false
            }
        } catch {
            print("Profile load error: \(error)")
        }
    }
}

// MARK: - Shared types

enum UserRole: String, CaseIterable {
    case coach   = "coach"
    case athlete = "athlete"
    case admin   = "admin"
    var displayName: String { rawValue.capitalized }
}

// MARK: - Private DB models

private struct ProfileRow: Decodable {
    let id: UUID
    let email: String?
    let fullName: String?
    let role: String?
    let profileComplete: Bool?
    let inviteCode: String?
    enum CodingKeys: String, CodingKey {
        case id, email, role
        case fullName       = "full_name"
        case profileComplete = "profile_complete"
        case inviteCode     = "invite_code"
    }
}
