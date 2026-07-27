import Foundation

public enum ScanValidationError: Error, Equatable, CustomStringConvertible {
    case missingItems
    case nonFiniteNumber(String)
    case negativeValue(String)
    case invalidConfidence(Double)
    case emptyName(index: Int)
    case nonPositiveGrams(index: Int)

    public var description: String {
        switch self {
        case .missingItems:
            return "items must not be empty"
        case let .nonFiniteNumber(field):
            return "\(field) must be finite"
        case let .negativeValue(field):
            return "\(field) must be >= 0"
        case let .invalidConfidence(value):
            return "confidence must be in [0,1], got \(value)"
        case let .emptyName(index):
            return "items[\(index)].name must not be empty"
        case let .nonPositiveGrams(index):
            return "items[\(index)].estimated_grams must be > 0"
        }
    }
}

public struct FoodScanRequest: Equatable, Sendable {
    public var imagePath: String
    public var mealType: MealType
    public var userHint: String?

    public init(imagePath: String, mealType: MealType, userHint: String? = nil) {
        self.imagePath = imagePath
        self.mealType = mealType
        self.userHint = userHint
    }
}

public enum MealType: String, Codable, CaseIterable, Sendable {
    case breakfast
    case lunch
    case dinner
    case snack
}

public enum UncertaintyFlag: String, Codable, Sendable {
    case portionUncertain = "portion_uncertain"
    case sauceHidden = "sauce_hidden"
    case mixedDish = "mixed_dish"
    case occlusion = "occlusion"
    case lowLight = "low_light"
}

public struct ScanItem: Codable, Equatable, Sendable {
    public var name: String
    public var estimatedGrams: Double
    public var calories: Double
    public var proteinGrams: Double
    public var fatGrams: Double
    public var carbsGrams: Double

    public init(name: String, estimatedGrams: Double, calories: Double, proteinGrams: Double, fatGrams: Double, carbsGrams: Double) {
        self.name = name
        self.estimatedGrams = estimatedGrams
        self.calories = calories
        self.proteinGrams = proteinGrams
        self.fatGrams = fatGrams
        self.carbsGrams = carbsGrams
    }

    enum CodingKeys: String, CodingKey {
        case name
        case estimatedGrams = "estimated_grams"
        case calories
        case proteinGrams = "protein_g"
        case fatGrams = "fat_g"
        case carbsGrams = "carbs_g"
    }
}

public struct FoodScanResult: Codable, Equatable, Sendable {
    public var totalCalories: Double
    public var proteinGrams: Double
    public var fatGrams: Double
    public var carbsGrams: Double
    /// Optional — not present in fine-tuning output (no ground truth in Nutrition5K).
    public var confidence: Double?
    public var items: [ScanItem]
    /// Optional — not present in fine-tuning output (no ground truth in Nutrition5K).
    public var uncertaintyFlags: [UncertaintyFlag]?

    public init(
        totalCalories: Double,
        proteinGrams: Double,
        fatGrams: Double,
        carbsGrams: Double,
        confidence: Double? = nil,
        items: [ScanItem],
        uncertaintyFlags: [UncertaintyFlag]? = nil
    ) {
        self.totalCalories = totalCalories
        self.proteinGrams = proteinGrams
        self.fatGrams = fatGrams
        self.carbsGrams = carbsGrams
        self.confidence = confidence
        self.items = items
        self.uncertaintyFlags = uncertaintyFlags
    }

    enum CodingKeys: String, CodingKey {
        case totalCalories = "total_calories"
        case proteinGrams = "protein_g"
        case fatGrams = "fat_g"
        case carbsGrams = "carbs_g"
        case confidence
        case items
        case uncertaintyFlags = "uncertainty_flags"
    }

    public func validated() throws -> FoodScanResult {
        guard !items.isEmpty else {
            throw ScanValidationError.missingItems
        }

        try Self.validateNonNegativeFinite(totalCalories, field: "total_calories")
        try Self.validateNonNegativeFinite(proteinGrams, field: "protein_g")
        try Self.validateNonNegativeFinite(fatGrams, field: "fat_g")
        try Self.validateNonNegativeFinite(carbsGrams, field: "carbs_g")

        if let confidence {
            guard confidence.isFinite else {
                throw ScanValidationError.nonFiniteNumber("confidence")
            }
            guard (0...1).contains(confidence) else {
                throw ScanValidationError.invalidConfidence(confidence)
            }
        }

        for (index, item) in items.enumerated() {
            if item.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                throw ScanValidationError.emptyName(index: index)
            }
            if !item.estimatedGrams.isFinite {
                throw ScanValidationError.nonFiniteNumber("items[\(index)].estimated_grams")
            }
            if item.estimatedGrams <= 0 {
                throw ScanValidationError.nonPositiveGrams(index: index)
            }
            try Self.validateNonNegativeFinite(item.calories, field: "items[\(index)].calories")
            try Self.validateNonNegativeFinite(item.proteinGrams, field: "items[\(index)].protein_g")
            try Self.validateNonNegativeFinite(item.fatGrams, field: "items[\(index)].fat_g")
            try Self.validateNonNegativeFinite(item.carbsGrams, field: "items[\(index)].carbs_g")
        }

        return self
    }

    private static func validateNonNegativeFinite(_ value: Double, field: String) throws {
        guard value.isFinite else {
            throw ScanValidationError.nonFiniteNumber(field)
        }
        guard value >= 0 else {
            throw ScanValidationError.negativeValue(field)
        }
    }
}

public enum ScanJSONParser {
    public static func parseStrict(from jsonData: Data) throws -> FoodScanResult {
        let decoder = JSONDecoder()
        let result = try decoder.decode(FoodScanResult.self, from: jsonData)
        return try result.validated()
    }

    public static func parseStrict(from jsonString: String) throws -> FoodScanResult {
        guard let data = jsonString.data(using: .utf8) else {
            throw NSError(domain: "ScanJSONParser", code: 1)
        }
        return try parseStrict(from: data)
    }

    /// True when the model returned the v7 not-food refusal — the object
    /// `{"not_food": true}` — instead of a nutrition result. This is a *correct*
    /// terminal answer for a non-food photo (a computer mouse, an empty plate),
    /// NOT a parse failure, so callers check it before `parseStrict` (which would
    /// throw `missingItems` on the empty-items refusal).
    ///
    /// Detection is lenient about surrounding prose: if the whole string isn't
    /// valid JSON, the first `{ ... }` block is probed for a top-level
    /// `not_food == true`.
    public static func isNotFood(_ jsonString: String) -> Bool {
        struct Probe: Decodable { let notFood: Bool?
            enum CodingKeys: String, CodingKey { case notFood = "not_food" }
        }
        for candidate in jsonObjectCandidates(in: jsonString) {
            if let data = candidate.data(using: .utf8),
               let probe = try? JSONDecoder().decode(Probe.self, from: data),
               probe.notFood == true {
                return true
            }
        }
        return false
    }

    /// The full string first, then the first brace-delimited substring — enough
    /// to recover `{"not_food": true}` if the model wrapped it in commentary.
    private static func jsonObjectCandidates(in text: String) -> [String] {
        var candidates = [text]
        if let open = text.firstIndex(of: "{"), let close = text.lastIndex(of: "}"), open < close {
            candidates.append(String(text[open...close]))
        }
        return candidates
    }
}
