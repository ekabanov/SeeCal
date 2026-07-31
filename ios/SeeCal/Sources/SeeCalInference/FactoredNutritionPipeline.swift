import Foundation
import SQLite3
import SeeCalDomain

public enum IdentifiedContainer: String, Codable, CaseIterable, Sendable {
    case plate, bowl, cup, tray, packaging, other
}

public struct IdentifiedFoodItem: Codable, Equatable, Sendable {
    public let name: String
    public let sharePercent: Int
    /// Side-channel confidence derived from name-token log probabilities. It is
    /// deliberately not part of the frozen model completion schema.
    public let nameConfidence: Double?

    public init(name: String, sharePercent: Int, nameConfidence: Double? = nil) {
        self.name = name
        self.sharePercent = sharePercent
        self.nameConfidence = nameConfidence
    }

    enum CodingKeys: String, CodingKey {
        case name
        case sharePercent = "share_pct"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        sharePercent = try container.decode(Int.self, forKey: .sharePercent)
        nameConfidence = nil
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(sharePercent, forKey: .sharePercent)
    }
}

public enum IdentificationValidationError: Error, Equatable, CustomStringConvertible {
    case unexpectedKeys
    case invalidContainer
    case invalidNotFood
    case missingItems
    case invalidName(index: Int)
    case invalidPortionUnits(index: Int)
    case invalidShare(index: Int)
    case duplicateName(String)
    case unsortedShares
    case invalidShareSum(Int)

    public var description: String {
        switch self {
        case .unexpectedKeys: return "IDENTIFY JSON contains unexpected or missing keys"
        case .invalidContainer: return "container is outside the frozen vocabulary"
        case .invalidNotFood: return "not-food requires container other and no items"
        case .missingItems: return "food identification requires at least one item"
        case let .invalidName(index): return "items[\(index)].name must be non-empty"
        case let .invalidPortionUnits(index):
            return "items[\(index)].portion_units must be an integer from 1 to 20"
        case let .invalidShare(index):
            return "items[\(index)].share_pct must be a positive multiple of 5"
        case let .duplicateName(name): return "duplicate item name: \(name)"
        case .unsortedShares: return "items must be sorted by share_pct descending"
        case let .invalidShareSum(sum): return "share_pct values must sum to 100, got \(sum)"
        }
    }
}

public struct FoodIdentification: Codable, Equatable, Sendable {
    public let notFood: Bool
    public let container: IdentifiedContainer
    public let items: [IdentifiedFoodItem]

    public init(
        notFood: Bool,
        container: IdentifiedContainer,
        items: [IdentifiedFoodItem]
    ) throws {
        self.notFood = notFood
        self.container = container
        self.items = items
        try validated()
    }

    enum CodingKeys: String, CodingKey {
        case notFood = "not_food"
        case container, items
    }

    public func validated() throws {
        if notFood {
            guard container == .other, items.isEmpty else {
                throw IdentificationValidationError.invalidNotFood
            }
            return
        }
        guard !items.isEmpty else { throw IdentificationValidationError.missingItems }
        var seen = Set<String>()
        var priorShare = 101
        var sum = 0
        for (index, item) in items.enumerated() {
            let normalized = FactoredNameNormalizer.normalize(item.name)
            guard !normalized.isEmpty else {
                throw IdentificationValidationError.invalidName(index: index)
            }
            guard item.sharePercent > 0, item.sharePercent.isMultiple(of: 5) else {
                throw IdentificationValidationError.invalidShare(index: index)
            }
            guard item.sharePercent <= priorShare else {
                throw IdentificationValidationError.unsortedShares
            }
            guard seen.insert(normalized).inserted else {
                throw IdentificationValidationError.duplicateName(item.name)
            }
            priorShare = item.sharePercent
            sum += item.sharePercent
        }
        guard sum == 100 else {
            throw IdentificationValidationError.invalidShareSum(sum)
        }
    }
}

public enum IdentificationJSONParser {
    private static func normalizedFood(
        container: IdentifiedContainer,
        weightedNames: [(name: String, weight: Double)]
    ) throws -> FoodIdentification {
        var merged: [String: (name: String, weight: Double)] = [:]
        for (index, item) in weightedNames.enumerated() {
            let normalized = FactoredNameNormalizer.normalize(item.name)
            guard !normalized.isEmpty else {
                throw IdentificationValidationError.invalidName(index: index)
            }
            if let existing = merged[normalized] {
                merged[normalized] = (existing.name, existing.weight + item.weight)
            } else {
                merged[normalized] = (
                    item.name.trimmingCharacters(in: .whitespacesAndNewlines),
                    item.weight
                )
            }
        }
        let values = Array(
            merged.values
                .sorted {
                    if $0.weight != $1.weight { return $0.weight > $1.weight }
                    return FactoredNameNormalizer.normalize($0.name)
                        < FactoredNameNormalizer.normalize($1.name)
                }
                .prefix(20)
        )
        let total = values.reduce(0) { $0 + $1.weight }
        let bucketCount = 20
        let exact = values.map { $0.weight / total * Double(bucketCount) }
        var allocated = Array(repeating: 1, count: values.count)
        let remaining = bucketCount - values.count
        if remaining > 0 {
            var desiredExtra = exact.map { max(0, $0 - 1) }
            var extraTotal = desiredExtra.reduce(0, +)
            if extraTotal == 0 {
                desiredExtra = exact
                extraTotal = desiredExtra.reduce(0, +)
            }
            let scaled = desiredExtra.map { $0 / extraTotal * Double(remaining) }
            let floors = scaled.map { Int(floor($0)) }
            for index in allocated.indices {
                allocated[index] += floors[index]
            }
            let leftovers = remaining - floors.reduce(0, +)
            let order = allocated.indices.sorted {
                let lhs = scaled[$0] - Double(floors[$0])
                let rhs = scaled[$1] - Double(floors[$1])
                return lhs == rhs ? $0 < $1 : lhs > rhs
            }
            for index in order.prefix(leftovers) {
                allocated[index] += 1
            }
        }
        return try FoodIdentification(
            notFood: false,
            container: container,
            items: zip(values, allocated).map {
                IdentifiedFoodItem(name: $0.0.name, sharePercent: $0.1 * 5)
            }
        )
    }

    public static func parseStrict(_ text: String) throws -> FoodIdentification {
        guard let data = text.data(using: .utf8) else {
            throw IdentificationValidationError.unexpectedKeys
        }
        let object = try JSONSerialization.jsonObject(with: data)
        guard let dictionary = object as? [String: Any],
              Set(dictionary.keys) == Set(["not_food", "container", "items"]),
              let rawItems = dictionary["items"] as? [[String: Any]],
              rawItems.allSatisfy({ Set($0.keys) == Set(["name", "portion_units"]) }),
              let notFood = dictionary["not_food"] as? Bool,
              let rawContainer = dictionary["container"] as? String,
              let identifiedContainer = IdentifiedContainer(rawValue: rawContainer)
        else {
            throw IdentificationValidationError.unexpectedKeys
        }
        if notFood {
            guard identifiedContainer == .other, rawItems.isEmpty else {
                throw IdentificationValidationError.invalidNotFood
            }
            return try FoodIdentification(
                notFood: true,
                container: identifiedContainer,
                items: []
            )
        }
        guard !rawItems.isEmpty else {
            throw IdentificationValidationError.missingItems
        }

        var weightedNames = [(name: String, weight: Double)]()
        for (index, rawItem) in rawItems.enumerated() {
            guard let name = rawItem["name"] as? String else {
                throw IdentificationValidationError.invalidName(index: index)
            }
            let normalized = FactoredNameNormalizer.normalize(name)
            guard !normalized.isEmpty else {
                throw IdentificationValidationError.invalidName(index: index)
            }
            guard let units = rawItem["portion_units"] as? Int,
                  (1 ... 20).contains(units)
            else {
                throw IdentificationValidationError.invalidPortionUnits(index: index)
            }
            weightedNames.append((name, Double(units)))
        }
        return try normalizedFood(
            container: identifiedContainer,
            weightedNames: weightedNames
        )
    }

    public static func parseLegacySharesNormalized(
        _ text: String
    ) throws -> FoodIdentification {
        guard let data = text.data(using: .utf8) else {
            throw IdentificationValidationError.unexpectedKeys
        }
        let object = try JSONSerialization.jsonObject(with: data)
        guard let dictionary = object as? [String: Any],
              Set(dictionary.keys) == Set(["not_food", "container", "items"]),
              let rawItems = dictionary["items"] as? [[String: Any]],
              rawItems.allSatisfy({ Set($0.keys) == Set(["name", "share_pct"]) }),
              let notFood = dictionary["not_food"] as? Bool,
              let rawContainer = dictionary["container"] as? String,
              let identifiedContainer = IdentifiedContainer(rawValue: rawContainer)
        else {
            throw IdentificationValidationError.unexpectedKeys
        }
        if notFood {
            guard identifiedContainer == .other, rawItems.isEmpty else {
                throw IdentificationValidationError.invalidNotFood
            }
            return try FoodIdentification(
                notFood: true,
                container: identifiedContainer,
                items: []
            )
        }
        guard !rawItems.isEmpty else {
            throw IdentificationValidationError.missingItems
        }
        let weightedNames: [(name: String, weight: Double)] = try rawItems.enumerated().map {
            index, rawItem in
            guard let name = rawItem["name"] as? String else {
                throw IdentificationValidationError.invalidName(index: index)
            }
            let normalized = FactoredNameNormalizer.normalize(name)
            guard !normalized.isEmpty else {
                throw IdentificationValidationError.invalidName(index: index)
            }
            guard let number = rawItem["share_pct"] as? NSNumber,
                  number.doubleValue.isFinite,
                  number.doubleValue > 0
            else {
                throw IdentificationValidationError.invalidShare(index: index)
            }
            return (name, number.doubleValue)
        }
        return try normalizedFood(
            container: identifiedContainer,
            weightedNames: weightedNames
        )
    }
}

public enum FactoredIdentificationPrompt {
    /// Byte-identical to `ml/factored_pipeline/contract.py`.
    public static let text =
        "Identify only visually distinguishable food components without estimating " +
        "grams, calories, nutrients, seasonings, or hidden recipe ingredients. " +
        "Return exactly one JSON object with keys not_food, container, and items. " +
        "container must be plate, bowl, cup, tray, packaging, or other. " +
        "items must contain objects with name and portion_units only. " +
        "portion_units must be an integer from 1 to 20 expressing relative visible " +
        "portion size; the values do not need to sum to any particular number. " +
        "For a non-food image return not_food true, container other, and an empty items list."

    /// The trained, byte-identical prompt remains untouched for normal scans.
    /// A hint is appended only after deterministic resolution explicitly asks
    /// the person for help. Whitespace is flattened and length is bounded so a
    /// broad description cannot accidentally consume the generation budget.
    public static func text(userHint: String?) -> String {
        guard let bounded = normalizedUserHint(userHint) else { return text }
        return text +
            "\n\nHuman-provided context (may be broad or imperfect): \(bounded). " +
            "Use it as supporting evidence while inspecting the image. Return the exact same JSON schema."
    }

    /// One normalization contract for recovery and proactive estimate fixes.
    /// Control characters become spaces before whitespace is collapsed; the
    /// resulting inference-only context is capped independently of UI widgets.
    public static func normalizedUserHint(_ userHint: String?) -> String? {
        guard let userHint else { return nil }
        let withoutControls = userHint.unicodeScalars
            .map { CharacterSet.controlCharacters.contains($0) ? " " : String($0) }
            .joined()
        let normalized = withoutControls
            .split(whereSeparator: \Character.isWhitespace)
            .joined(separator: " ")
        guard !normalized.isEmpty else { return nil }
        return String(normalized.prefix(240))
    }

    /// Byte-identical historical prompt for the already-trained percentage adapter.
    public static let legacyPercentageText =
        "Identify the visible food without estimating grams, calories, or nutrients. " +
        "Return exactly one JSON object with keys not_food, container, and items. " +
        "container must be plate, bowl, cup, tray, packaging, or other. " +
        "items must contain objects with name and share_pct only, sorted by share_pct " +
        "descending. share_pct values must be multiples of 5 and sum to 100. " +
        "For a non-food image return not_food true, container other, and an empty items list."
}

public struct FactoredScalePrediction: Equatable, Sendable {
    public let massGrams: VisualSpecialistInterval

    public init(massGrams: VisualSpecialistInterval) throws {
        guard massGrams.low.isFinite,
              massGrams.estimate.isFinite,
              massGrams.high.isFinite,
              massGrams.low >= 0,
              massGrams.low <= massGrams.estimate,
              massGrams.estimate <= massGrams.high
        else {
            throw FactoredPipelineError.invalidScaleInterval
        }
        self.massGrams = massGrams
    }
}

public struct ResolvedNutritionProfile: Equatable, Sendable {
    public let fdcID: Int64?
    public let name: String
    public let category: String
    public let kcalPer100g: Double
    public let proteinPer100g: Double
    public let fatPer100g: Double
    public let carbsPer100g: Double
    public let sourceKcalPer100g: Double?
    public let dataType: String
    public let typicalPortionGrams: Double?

    public init(
        fdcID: Int64?,
        name: String,
        category: String,
        kcalPer100g: Double,
        proteinPer100g: Double,
        fatPer100g: Double,
        carbsPer100g: Double,
        sourceKcalPer100g: Double? = nil,
        dataType: String,
        typicalPortionGrams: Double? = nil
    ) throws {
        let values = [kcalPer100g, proteinPer100g, fatPer100g, carbsPer100g]
        guard values.allSatisfy({ $0.isFinite && $0 >= 0 }),
              kcalPer100g <= 900,
              proteinPer100g <= 100,
              fatPer100g <= 100,
              carbsPer100g <= 100
        else {
            throw FactoredPipelineError.unsafeNutritionProfile(name)
        }
        self.fdcID = fdcID
        self.name = name
        self.category = category
        self.kcalPer100g = kcalPer100g
        self.proteinPer100g = proteinPer100g
        self.fatPer100g = fatPer100g
        self.carbsPer100g = carbsPer100g
        self.sourceKcalPer100g = sourceKcalPer100g
        self.dataType = dataType
        self.typicalPortionGrams = typicalPortionGrams
    }
}

public enum NutritionResolutionRung: String, Equatable, Sendable {
    case exactAlias = "exact_alias"
    case fuzzy
    case categoryDefault = "category_default"
    case hypothesis
    case unresolved
}

public struct NutritionResolution: Equatable, Sendable {
    public let query: String
    public let rung: NutritionResolutionRung
    public let score: Double
    public let profile: ResolvedNutritionProfile?
    public let estimated: Bool

    public init(
        query: String,
        rung: NutritionResolutionRung,
        score: Double,
        profile: ResolvedNutritionProfile?,
        estimated: Bool
    ) {
        self.query = query
        self.rung = rung
        self.score = score
        self.profile = profile
        self.estimated = estimated
    }
}

public protocol NutritionResolving: Sendable {
    func resolve(name: String) async throws -> NutritionResolution
}

/// A database-backed choice shown when the user wants to replace a recognized
/// food. Values are expressed per 100 g so the editor can preserve the current
/// amount while swapping the nutrition density.
public struct NutritionProfileCandidate: Equatable, Sendable, Identifiable {
    public let profile: ResolvedNutritionProfile
    public let displayName: String
    public let score: Double

    public init(
        profile: ResolvedNutritionProfile,
        displayName: String? = nil,
        score: Double
    ) {
        self.profile = profile
        self.displayName = displayName ?? profile.name
        self.score = score
    }

    public var id: String {
        profile.fdcID.map(String.init)
            ?? "\(profile.dataType):\(FactoredNameNormalizer.normalize(profile.name))"
    }
}

public protocol NutritionCandidateProviding: Sendable {
    func candidates(for name: String, limit: Int) async -> [NutritionProfileCandidate]
}

public protocol NutritionHypothesisProviding: Sendable {
    func hypothesis(
        name: String,
        category: String?
    ) async throws -> ResolvedNutritionProfile?
}

public struct NoNutritionHypothesisProvider: NutritionHypothesisProviding {
    public init() {}

    public func hypothesis(
        name: String,
        category: String?
    ) async throws -> ResolvedNutritionProfile? {
        nil
    }
}

public enum NutritionDatabaseError: Error, CustomStringConvertible {
    case openFailed(String)
    case queryFailed(String)
    case malformedRow

    public var description: String {
        switch self {
        case let .openFailed(reason): return "Could not open nutrition database: \(reason)"
        case let .queryFailed(reason): return "Nutrition database query failed: \(reason)"
        case .malformedRow: return "Nutrition database contains a malformed row"
        }
    }
}

public final class SQLiteNutritionResolver:
    NutritionResolving,
    NutritionCandidateProviding,
    @unchecked Sendable
{
    private struct Candidate: Sendable {
        let normalizedName: String
        let profile: ResolvedNutritionProfile
    }

    private let exactAliases: [String: ResolvedNutritionProfile]
    private let foods: [Candidate]
    private let tokenIndex: [String: [Candidate]]
    private let prefixIndex: [String: [Candidate]]
    private let compactIndex: [String: [Candidate]]
    private let categoryDefaults: [String: ResolvedNutritionProfile]
    private let fuzzyThreshold: Double
    private let categoryThreshold: Double
    private let hypothesisProvider: any NutritionHypothesisProviding

    public init(
        databaseURL: URL,
        fuzzyThreshold: Double = 0.76,
        categoryThreshold: Double = 0.42,
        hypothesisProvider: any NutritionHypothesisProviding = NoNutritionHypothesisProvider()
    ) throws {
        var database: OpaquePointer?
        guard sqlite3_open_v2(
            databaseURL.path,
            &database,
            SQLITE_OPEN_READONLY,
            nil
        ) == SQLITE_OK, let database else {
            let message = database.map { String(cString: sqlite3_errmsg($0)) } ?? "unknown error"
            if let database { sqlite3_close(database) }
            throw NutritionDatabaseError.openFailed(message)
        }
        defer { sqlite3_close(database) }
        let foods = try Self.loadFoods(database)
        self.foods = foods
        let indexes = Self.buildIndexes(foods)
        self.tokenIndex = indexes.tokens
        self.prefixIndex = indexes.prefixes
        self.compactIndex = indexes.compact
        self.categoryDefaults = try Self.loadCategoryDefaults(database)
        self.exactAliases = try Self.loadAliases(database, foods: foods)
        self.fuzzyThreshold = fuzzyThreshold
        self.categoryThreshold = categoryThreshold
        self.hypothesisProvider = hypothesisProvider
    }

    public func resolve(name: String) async throws -> NutritionResolution {
        let normalized = FactoredNameNormalizer.normalize(name)
        guard !normalized.isEmpty else {
            return .init(query: name, rung: .unresolved, score: 0, profile: nil, estimated: true)
        }
        if let exact = exactAliases[normalized] {
            return .init(query: name, rung: .exactAlias, score: 1, profile: exact, estimated: false)
        }
        let compact = normalized.replacingOccurrences(of: " ", with: "")
        if let compactMatch = compactIndex[compact]?.first {
            return .init(
                query: name,
                rung: .fuzzy,
                score: 1,
                profile: compactMatch.profile,
                estimated: false
            )
        }
        var candidateMap: [Int64: Candidate] = [:]
        for token in FactoredNameNormalizer.tokens(normalized) {
            for candidate in tokenIndex[token] ?? [] {
                if let fdcID = candidate.profile.fdcID {
                    candidateMap[fdcID] = candidate
                }
            }
        }
        let prefix = String(compact.prefix(2))
        let candidates = candidateMap.isEmpty
            ? (prefixIndex[prefix] ?? foods)
            : candidateMap.values.sorted {
                if $0.normalizedName != $1.normalizedName {
                    return $0.normalizedName < $1.normalizedName
                }
                return ($0.profile.fdcID ?? 0) < ($1.profile.fdcID ?? 0)
            }
        let best = candidates
            .map { ($0, FactoredNameNormalizer.lexicalScore(normalized, $0.normalizedName)) }
            .max { $0.1 < $1.1 }
        if let best, best.1 >= fuzzyThreshold {
            return .init(query: name, rung: .fuzzy, score: best.1, profile: best.0.profile, estimated: false)
        }
        var scoresByCategory: [String: [Double]] = [:]
        for food in candidates {
            let score = FactoredNameNormalizer.categoryLexicalScore(
                normalized,
                food.normalizedName
            )
            if score > 0 {
                scoresByCategory[food.profile.category, default: []].append(score)
            }
        }
        let categoryChoice = scoresByCategory.map { category, scores in
            let ranked = scores.sorted(by: >)
            return (
                category: category,
                clusterScore: ranked.prefix(5).reduce(0, +),
                bestScore: ranked.first ?? 0
            )
        }.max {
            if $0.clusterScore != $1.clusterScore {
                return $0.clusterScore < $1.clusterScore
            }
            if $0.bestScore != $1.bestScore {
                return $0.bestScore < $1.bestScore
            }
            return $0.category < $1.category
        }
        let category = categoryChoice.flatMap {
            $0.bestScore >= categoryThreshold ? $0.category : nil
        }
        if let category, let profile = categoryDefaults[category] {
            return .init(
                query: name,
                rung: .categoryDefault,
                score: categoryChoice?.bestScore ?? 0,
                profile: profile,
                estimated: true
            )
        }
        if let hypothesis = try await hypothesisProvider.hypothesis(name: name, category: category) {
            return .init(
                query: name,
                rung: .hypothesis,
                score: best?.1 ?? 0,
                profile: hypothesis,
                estimated: true
            )
        }
        return .init(
            query: name,
            rung: .unresolved,
            score: best?.1 ?? 0,
            profile: nil,
            estimated: true
        )
    }

    /// Returns a small, deterministic replacement set. Common visual
    /// confusions are seeded first, then nearby database foods fill any open
    /// slots. The current food is excluded.
    public func candidates(
        for name: String,
        limit: Int = 5
    ) async -> [NutritionProfileCandidate] {
        guard limit > 0 else { return [] }
        let normalized = FactoredNameNormalizer.normalize(name)
        guard !normalized.isEmpty else { return [] }

        let current = profileForCandidateQuery(normalized)
        var output: [NutritionProfileCandidate] = []
        var seen = Set<String>()

        func append(
            _ profile: ResolvedNutritionProfile,
            displayName: String? = nil,
            score: Double
        ) {
            let key = profile.fdcID.map(String.init)
                ?? "\(profile.dataType):\(FactoredNameNormalizer.normalize(profile.name))"
            guard key != current.map(profileKey), seen.insert(key).inserted else { return }
            output.append(.init(profile: profile, displayName: displayName, score: score))
        }

        for (index, seed) in Self.replacementSeeds(for: normalized).enumerated() {
            if let profile = profileForCandidateQuery(seed, allowLooseTokenMatch: true) {
                append(
                    profile,
                    displayName: seed.capitalized,
                    score: 1 - Double(index) * 0.01
                )
            }
        }
        if output.count >= limit {
            return Array(output.prefix(limit))
        }

        let currentCategory = current?.category
        var indexedCandidates: [Int64: Candidate] = [:]
        for token in FactoredNameNormalizer.tokens(normalized) {
            for candidate in tokenIndex[token] ?? [] {
                if let fdcID = candidate.profile.fdcID {
                    indexedCandidates[fdcID] = candidate
                }
            }
        }
        let compact = normalized.replacingOccurrences(of: " ", with: "")
        let candidatePool: [Candidate]
        if !indexedCandidates.isEmpty {
            candidatePool = Array(indexedCandidates.values)
        } else if let prefixCandidates = prefixIndex[String(compact.prefix(2))],
                  !prefixCandidates.isEmpty {
            candidatePool = prefixCandidates
        } else {
            candidatePool = foods
        }
        let ranked = candidatePool.compactMap { candidate -> (Candidate, Double)? in
            let lexical = FactoredNameNormalizer.lexicalScore(
                normalized,
                candidate.normalizedName
            )
            let categoryBoost = candidate.profile.category == currentCategory ? 0.18 : 0
            let score = lexical + categoryBoost
            return score > 0 ? (candidate, score) : nil
        }.sorted {
            if $0.1 != $1.1 { return $0.1 > $1.1 }
            if $0.0.normalizedName != $1.0.normalizedName {
                return $0.0.normalizedName < $1.0.normalizedName
            }
            return ($0.0.profile.fdcID ?? 0) < ($1.0.profile.fdcID ?? 0)
        }
        for (candidate, score) in ranked {
            append(candidate.profile, score: score)
            if output.count >= limit { break }
        }
        return Array(output.prefix(limit))
    }

    private func profileForCandidateQuery(
        _ query: String,
        allowLooseTokenMatch: Bool = false
    ) -> ResolvedNutritionProfile? {
        let normalized = FactoredNameNormalizer.normalize(query)
        if let exact = exactAliases[normalized] {
            return exact
        }
        let compact = normalized.replacingOccurrences(of: " ", with: "")
        if let exact = compactIndex[compact]?.first?.profile {
            return exact
        }
        var indexedCandidates: [Int64: Candidate] = [:]
        for token in FactoredNameNormalizer.tokens(normalized) {
            for candidate in tokenIndex[token] ?? [] {
                if let fdcID = candidate.profile.fdcID {
                    indexedCandidates[fdcID] = candidate
                }
            }
        }
        let candidatePool = indexedCandidates.isEmpty
            ? foods
            : Array(indexedCandidates.values)
        let best = candidatePool
            .map { candidate in
                let lexical = FactoredNameNormalizer.lexicalScore(
                    normalized,
                    candidate.normalizedName
                )
                let startsWithBoost = candidate.normalizedName.hasPrefix(normalized) ? 0.1 : 0
                let specificityPenalty =
                    Double(FactoredNameNormalizer.tokens(candidate.normalizedName).count) * 0.002
                return (
                    candidate.profile,
                    lexical + startsWithBoost - specificityPenalty
                )
            }
            .max { $0.1 < $1.1 }
        let threshold = allowLooseTokenMatch ? min(fuzzyThreshold, 0.42) : fuzzyThreshold
        return best.flatMap { $0.1 >= threshold ? $0.0 : nil }
    }

    private func profileKey(_ profile: ResolvedNutritionProfile) -> String {
        profile.fdcID.map(String.init)
            ?? "\(profile.dataType):\(FactoredNameNormalizer.normalize(profile.name))"
    }

    /// Small, transparent priors for frequent photo-level confusions. These
    /// only choose among real local database rows; they never invent nutrition.
    private static func replacementSeeds(for normalized: String) -> [String] {
        let groups: [[String]] = [
            ["chicken", "turkey", "tofu", "salmon", "egg", "paneer"],
            ["beef", "pork", "chicken", "lamb", "tofu", "mushroom"],
            ["salmon", "tuna", "white fish", "shrimp", "chicken", "tofu"],
            ["rice", "quinoa", "pasta", "potato", "couscous", "cauliflower"],
            ["broccoli", "green beans", "asparagus", "spinach", "zucchini", "peas"],
            ["potato", "sweet potato", "rice", "pasta", "bread", "quinoa"],
            ["yogurt", "cottage cheese", "sour cream", "cream cheese", "tofu"],
            ["egg", "tofu", "chicken", "sausage", "yogurt"],
        ]
        let tokens = Set(FactoredNameNormalizer.tokens(normalized))
        guard let group = groups.first(where: { candidates in
            candidates.contains { seed in
                tokens.contains(where: { seed.contains($0) || $0.contains(seed) })
            }
        }) else {
            return []
        }
        return group.filter { seed in
            !tokens.contains(where: { seed.contains($0) || $0.contains(seed) })
        }
    }

    private static let foodColumns =
        "fdc_id,name,normalized_name,category,data_type,kcal_per_100g," +
        "source_kcal_per_100g,protein_per_100g,fat_per_100g,carbs_per_100g," +
        "typical_portion_g"

    private static func loadFoods(_ database: OpaquePointer) throws -> [Candidate] {
        try rows(
            database,
            sql: "SELECT \(foodColumns) FROM foods ORDER BY normalized_name,fdc_id"
        ).map { row in
            Candidate(
                normalizedName: row.normalizedName,
                profile: row.profile
            )
        }
    }

    private static func buildIndexes(
        _ foods: [Candidate]
    ) -> (
        tokens: [String: [Candidate]],
        prefixes: [String: [Candidate]],
        compact: [String: [Candidate]]
    ) {
        var tokens: [String: [Candidate]] = [:]
        var prefixes: [String: [Candidate]] = [:]
        var compact: [String: [Candidate]] = [:]
        for candidate in foods {
            let compactName = candidate.normalizedName.replacingOccurrences(
                of: " ",
                with: ""
            )
            compact[compactName, default: []].append(candidate)
            prefixes[String(compactName.prefix(2)), default: []].append(candidate)
            for token in FactoredNameNormalizer.tokens(candidate.normalizedName) {
                if token.count >= 2 {
                    tokens[token, default: []].append(candidate)
                }
            }
        }
        return (tokens, prefixes, compact)
    }

    private static func loadAliases(
        _ database: OpaquePointer,
        foods: [Candidate]
    ) throws -> [String: ResolvedNutritionProfile] {
        let profiles = Dictionary(
            uniqueKeysWithValues: foods.compactMap { candidate in
                candidate.profile.fdcID.map { ($0, candidate.profile) }
            }
        )
        var statement: OpaquePointer?
        let sql =
            "SELECT normalized_alias,fdc_id FROM aliases " +
            "ORDER BY normalized_alias,priority DESC,fdc_id"
        guard sqlite3_prepare_v2(database, sql, -1, &statement, nil) == SQLITE_OK,
              let statement
        else {
            throw NutritionDatabaseError.queryFailed(String(cString: sqlite3_errmsg(database)))
        }
        defer { sqlite3_finalize(statement) }
        var aliases: [String: ResolvedNutritionProfile] = [:]
        while sqlite3_step(statement) == SQLITE_ROW {
            guard let aliasPointer = sqlite3_column_text(statement, 0) else {
                throw NutritionDatabaseError.malformedRow
            }
            let alias = String(cString: aliasPointer)
            let fdcID = sqlite3_column_int64(statement, 1)
            if aliases[alias] == nil {
                aliases[alias] = profiles[fdcID]
            }
        }
        return aliases
    }

    private static func loadCategoryDefaults(
        _ database: OpaquePointer
    ) throws -> [String: ResolvedNutritionProfile] {
        let sql =
            "SELECT NULL AS fdc_id,name,normalized_name,category,data_type," +
            "kcal_per_100g,source_kcal_per_100g,protein_per_100g,fat_per_100g," +
            "carbs_per_100g,typical_portion_g FROM category_defaults ORDER BY category"
        return Dictionary(uniqueKeysWithValues: try rows(database, sql: sql).map {
            ($0.profile.category, $0.profile)
        })
    }

    private struct DatabaseRow {
        let normalizedName: String
        let profile: ResolvedNutritionProfile
    }

    private static func rows(
        _ database: OpaquePointer,
        sql: String
    ) throws -> [DatabaseRow] {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(database, sql, -1, &statement, nil) == SQLITE_OK,
              let statement
        else {
            throw NutritionDatabaseError.queryFailed(String(cString: sqlite3_errmsg(database)))
        }
        defer { sqlite3_finalize(statement) }
        func text(_ index: Int32) throws -> String {
            guard let pointer = sqlite3_column_text(statement, index) else {
                throw NutritionDatabaseError.malformedRow
            }
            return String(cString: pointer)
        }
        func optionalDouble(_ index: Int32) -> Double? {
            sqlite3_column_type(statement, index) == SQLITE_NULL
                ? nil
                : sqlite3_column_double(statement, index)
        }
        var output: [DatabaseRow] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            let profile = try ResolvedNutritionProfile(
                fdcID: sqlite3_column_type(statement, 0) == SQLITE_NULL
                    ? nil
                    : sqlite3_column_int64(statement, 0),
                name: try text(1),
                category: try text(3),
                kcalPer100g: sqlite3_column_double(statement, 5),
                proteinPer100g: sqlite3_column_double(statement, 7),
                fatPer100g: sqlite3_column_double(statement, 8),
                carbsPer100g: sqlite3_column_double(statement, 9),
                sourceKcalPer100g: optionalDouble(6),
                dataType: try text(4),
                typicalPortionGrams: optionalDouble(10)
            )
            output.append(DatabaseRow(normalizedName: try text(2), profile: profile))
        }
        return output
    }
}

public struct FactoredNutritionInterval: Equatable, Sendable {
    public let estimate: Double
    public let low: Double
    public let high: Double
}

public struct FactoredNutritionValues: Equatable, Sendable {
    public let calories: FactoredNutritionInterval
    public let proteinGrams: FactoredNutritionInterval
    public let fatGrams: FactoredNutritionInterval
    public let carbsGrams: FactoredNutritionInterval
}

public struct FactoredMealItem: Equatable, Sendable {
    public let identification: IdentifiedFoodItem
    public let resolution: NutritionResolution
    public let grams: FactoredNutritionInterval
    /// Nil is a visible unresolved state, never a fabricated all-zero profile.
    public let nutrition: FactoredNutritionValues?
}

public enum FactoredConfirmationReason: String, Equatable, Sendable {
    case unresolvedItem
    case estimatedResolution
    case lowIdentificationConfidence
    case scaleDisagreement
}

public struct FactoredMeal: Equatable, Sendable {
    public let container: IdentifiedContainer
    public let scale: FactoredScalePrediction
    public let items: [FactoredMealItem]
    public let confirmationReasons: Set<FactoredConfirmationReason>

    public var foodScanResult: FoodScanResult? {
        guard items.allSatisfy({ $0.nutrition != nil }) else { return nil }
        let scanItems = items.compactMap { item -> ScanItem? in
            guard let nutrition = item.nutrition else { return nil }
            return ScanItem(
                name: item.identification.name,
                estimatedGrams: item.grams.estimate,
                calories: nutrition.calories.estimate,
                proteinGrams: nutrition.proteinGrams.estimate,
                fatGrams: nutrition.fatGrams.estimate,
                carbsGrams: nutrition.carbsGrams.estimate
            )
        }
        let totals = scanItems.reduce(
            (calories: 0.0, protein: 0.0, fat: 0.0, carbs: 0.0)
        ) { partial, item in
            (
                partial.calories + item.calories,
                partial.protein + item.proteinGrams,
                partial.fat + item.fatGrams,
                partial.carbs + item.carbsGrams
            )
        }
        return FoodScanResult(
            totalCalories: totals.calories,
            proteinGrams: totals.protein,
            fatGrams: totals.fat,
            carbsGrams: totals.carbs,
            items: scanItems
        )
    }
}

public enum FactoredPipelineError: Error, Equatable {
    case notFood
    case invalidScaleInterval
    case unsafeNutritionProfile(String)
    case resolutionCountMismatch
}

public enum FactoredAssembler {
    public static func assemble(
        identification: FoodIdentification,
        scale: FactoredScalePrediction,
        resolutions: [NutritionResolution],
        lowConfidenceThreshold: Double = 0.55
    ) throws -> FactoredMeal {
        try identification.validated()
        guard !identification.notFood else { throw FactoredPipelineError.notFood }
        guard resolutions.count == identification.items.count else {
            throw FactoredPipelineError.resolutionCountMismatch
        }
        var reasons = Set<FactoredConfirmationReason>()
        var portionPriorMass = 0.0
        var hasPortionPrior = false
        let items = zip(identification.items, resolutions).map { item, resolution in
            let share = Double(item.sharePercent) / 100
            let grams = FactoredNutritionInterval(
                estimate: share * scale.massGrams.estimate,
                low: share * scale.massGrams.low,
                high: share * scale.massGrams.high
            )
            if let confidence = item.nameConfidence, confidence < lowConfidenceThreshold {
                reasons.insert(.lowIdentificationConfidence)
            }
            guard let profile = resolution.profile else {
                reasons.insert(.unresolvedItem)
                return FactoredMealItem(
                    identification: item,
                    resolution: resolution,
                    grams: grams,
                    nutrition: nil
                )
            }
            if resolution.estimated {
                reasons.insert(.estimatedResolution)
            }
            if let portion = profile.typicalPortionGrams, portion > 0 {
                // The portion prior is independent evidence only. Sum one
                // selected serving per resolved item; do not infer total mass
                // by dividing each portion by its predicted share and then
                // averaging incompatible serving kinds.
                portionPriorMass += portion
                hasPortionPrior = true
            }
            func interval(_ per100g: Double) -> FactoredNutritionInterval {
                FactoredNutritionInterval(
                    estimate: grams.estimate / 100 * per100g,
                    low: grams.low / 100 * per100g,
                    high: grams.high / 100 * per100g
                )
            }
            return FactoredMealItem(
                identification: item,
                resolution: resolution,
                grams: grams,
                nutrition: FactoredNutritionValues(
                    calories: interval(profile.kcalPer100g),
                    proteinGrams: interval(profile.proteinPer100g),
                    fatGrams: interval(profile.fatPer100g),
                    carbsGrams: interval(profile.carbsPer100g)
                )
            )
        }
        if hasPortionPrior,
           portionPriorMass < scale.massGrams.low * 0.5
            || portionPriorMass > scale.massGrams.high * 2 {
            reasons.insert(.scaleDisagreement)
        }
        return FactoredMeal(
            container: identification.container,
            scale: scale,
            items: items,
            confirmationReasons: reasons
        )
    }

}

public enum FactoredShadowAdapter {
    public static func identification(from result: FoodScanResult) throws -> FoodIdentification {
        let items = quantizedShares(
            result.items.map { ($0.name, max(0, $0.estimatedGrams)) }
        )
        return try FoodIdentification(notFood: false, container: .other, items: items)
    }

    public static func scale(
        from specialist: VisualSpecialistPrediction
    ) throws -> FactoredScalePrediction {
        try FactoredScalePrediction(massGrams: specialist.massG)
    }

    private static func quantizedShares(
        _ weightedNames: [(String, Double)]
    ) -> [IdentifiedFoodItem] {
        let ranked = weightedNames
            .filter { !$0.0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.1 > 0 }
            .sorted { $0.1 > $1.1 }
            .prefix(20)
        let total = ranked.reduce(0) { $0 + $1.1 }
        guard total > 0 else { return [] }
        let exact = ranked.map { $0.1 / total * 20 }
        var buckets = Array(repeating: 1, count: ranked.count)
        var remaining = 20 - ranked.count
        while remaining > 0 {
            let index = buckets.indices.max {
                (exact[$0] - Double(buckets[$0])) < (exact[$1] - Double(buckets[$1]))
            }!
            buckets[index] += 1
            remaining -= 1
        }
        return zip(ranked, buckets)
            .map { IdentifiedFoodItem(name: $0.0.0, sharePercent: $0.1 * 5) }
            .sorted {
                if $0.sharePercent != $1.sharePercent {
                    return $0.sharePercent > $1.sharePercent
                }
                return $0.name < $1.name
            }
    }
}

public enum FactoredShadowPipeline {
    public static func run(
        monolithResult: FoodScanResult,
        specialistResult: VisualSpecialistPrediction,
        resolver: any NutritionResolving
    ) async throws -> FactoredMeal {
        let identification = try FactoredShadowAdapter.identification(from: monolithResult)
        let scale = try FactoredShadowAdapter.scale(from: specialistResult)
        var resolutions: [NutritionResolution] = []
        for item in identification.items {
            resolutions.append(try await resolver.resolve(name: item.name))
        }
        return try FactoredAssembler.assemble(
            identification: identification,
            scale: scale,
            resolutions: resolutions
        )
    }
}

private enum FactoredNameNormalizer {
    static func normalize(_ value: String) -> String {
        let folded = value
            .folding(options: [.diacriticInsensitive, .caseInsensitive], locale: .init(identifier: "en_US_POSIX"))
            .replacingOccurrences(of: "&", with: " and ")
        let allowed = folded.unicodeScalars.map { scalar -> Character in
            CharacterSet.alphanumerics.contains(scalar) ? Character(String(scalar)) : " "
        }
        return String(allowed).split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    static func lexicalScore(_ lhs: String, _ rhs: String) -> Double {
        let left = normalize(lhs)
        let right = normalize(rhs)
        guard !left.isEmpty, !right.isEmpty else { return 0 }
        let compactLeft = left.replacingOccurrences(of: " ", with: "")
        let compactRight = right.replacingOccurrences(of: " ", with: "")
        if compactLeft == compactRight { return 1 }
        let leftTokens = tokens(left)
        let rightTokens = tokens(right)
        if leftTokens == rightTokens { return 0.98 }
        let union = leftTokens.union(rightTokens)
        let overlap = union.isEmpty ? 0 : Double(leftTokens.intersection(rightTokens).count) / Double(union.count)
        let edit = 1 - Double(levenshtein(compactLeft, compactRight)) / Double(max(compactLeft.count, compactRight.count))
        let containment = compactLeft.contains(compactRight) || compactRight.contains(compactLeft) ? 1.0 : 0.0
        return 0.55 * overlap + 0.35 * edit + 0.10 * containment
    }

    static func categoryLexicalScore(_ lhs: String, _ rhs: String) -> Double {
        let left = normalize(lhs)
        let right = normalize(rhs)
        guard !left.isEmpty, !right.isEmpty else { return 0 }
        let leftTokens = tokens(left)
        let rightTokens = tokens(right)
        let intersection = leftTokens.intersection(rightTokens).count
        let coverage = Double(intersection) / Double(leftTokens.count)
        let union = leftTokens.union(rightTokens)
        let overlap = union.isEmpty ? 0 : Double(intersection) / Double(union.count)
        let edit =
            1 - Double(levenshtein(left, right)) / Double(max(left.count, right.count))
        return 0.70 * coverage + 0.20 * overlap + 0.10 * edit
    }

    static func tokens(_ value: String) -> Set<String> {
        Set(normalize(value).split(separator: " ").map {
            singularToken(String($0))
        })
    }

    private static func singularToken(_ token: String) -> String {
        if token.count > 4, token.hasSuffix("ies") {
            return String(token.dropLast(3)) + "y"
        }
        if token.count > 4,
           ["ches", "shes", "xes", "zes"].contains(where: { token.hasSuffix($0) })
        {
            return String(token.dropLast(2))
        }
        if token.count > 3, token.hasSuffix("s"), !token.hasSuffix("ss") {
            return String(token.dropLast())
        }
        return token
    }

    private static func levenshtein(_ lhs: String, _ rhs: String) -> Int {
        let left = Array(lhs), right = Array(rhs)
        var prior = Array(0...right.count)
        for (leftIndex, leftCharacter) in left.enumerated() {
            var current = [leftIndex + 1]
            for (rightIndex, rightCharacter) in right.enumerated() {
                current.append(
                    min(
                        current[rightIndex] + 1,
                        prior[rightIndex + 1] + 1,
                        prior[rightIndex] + (leftCharacter == rightCharacter ? 0 : 1)
                    )
                )
            }
            prior = current
        }
        return prior.last ?? 0
    }
}
