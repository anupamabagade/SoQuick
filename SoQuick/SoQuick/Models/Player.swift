import Foundation

struct Player: Codable, Identifiable {
    let id: UUID
    var userId: UUID?
    var addedBy: UUID?
    var name: String
    var age: Int?
    var heightFeet: Int?
    var heightInches: Int?
    var weightLbs: Double?
    var handedness: String?
    var levelOfPlay: String?
    var fastestPitch: Double?
    var zipCode: String?

    enum CodingKeys: String, CodingKey {
        case id
        case userId       = "user_id"
        case addedBy      = "added_by"
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

enum LevelOfPlay: String, CaseIterable {
    case recreation       = "Recreation"
    case travel10U        = "Travel/Club 10U"
    case travel12U        = "Travel/Club 12U"
    case travel14U        = "Travel/Club 14U"
    case travel16U        = "Travel/Club 16U"
    case travel18U        = "Travel/Club 18U"
    case juniorCollege    = "Junior College"
    case fourYearCollege  = "4-Year College/University"
    case professional     = "Professional"

    var display: String { rawValue }
}
