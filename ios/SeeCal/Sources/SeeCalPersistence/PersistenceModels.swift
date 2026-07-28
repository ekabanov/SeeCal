import Foundation
import SeeCalDomain

public enum MealOrigin: String, Codable, Equatable, Sendable {
    case photo
    case manual
    case barcode
}

/// Provenance retained with a barcode meal. Nutrition itself is snapshotted in
/// the item's immutable `sourceReference`, so old logs never change when the
/// community record changes.
public struct BarcodeSourceMetadata: Codable, Equatable, Sendable {
    public var barcode: String
    public var provider: String
    public var lookedUpAt: Date
    public var productName: String
    public var ingredientsText: String?
    public var servingDescription: String?

    public init(
        barcode: String,
        provider: String,
        lookedUpAt: Date,
        productName: String,
        ingredientsText: String? = nil,
        servingDescription: String? = nil
    ) {
        self.barcode = barcode
        self.provider = provider
        self.lookedUpAt = lookedUpAt
        self.productName = productName
        self.ingredientsText = ingredientsText
        self.servingDescription = servingDescription
    }
}

/// A logged (or being-logged) meal. Per spec §2, nutrition totals are never stored
/// independently — they are always the sum of `items`' scaled values (see
/// `MealLogEntry.totals`). `volumeMl`/`maxHeightMm` are depth metadata, populated
/// once D5 (LiDAR portion measurement) lands; `nil` until then.
public struct MealLogEntry: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var name: String
    public var createdAt: Date
    public var mealType: MealType
    public var imagePath: String?
    public var origin: MealOrigin
    public var barcodeSource: BarcodeSourceMetadata?
    public var items: [MealItem]
    public var volumeMl: Double?
    public var maxHeightMm: Double?

    public init(
        id: UUID = UUID(),
        name: String? = nil,
        createdAt: Date = Date(),
        mealType: MealType,
        imagePath: String? = nil,
        origin: MealOrigin? = nil,
        barcodeSource: BarcodeSourceMetadata? = nil,
        items: [MealItem],
        volumeMl: Double? = nil,
        maxHeightMm: Double? = nil
    ) {
        self.id = id
        self.createdAt = createdAt
        self.mealType = mealType
        self.imagePath = imagePath
        self.origin = origin ?? (barcodeSource == nil ? (imagePath == nil ? .manual : .photo) : .barcode)
        self.barcodeSource = barcodeSource
        self.items = items
        self.volumeMl = volumeMl
        self.maxHeightMm = maxHeightMm
        self.name = name ?? Self.defaultName(items: items, mealType: mealType)
    }

    /// Entry totals are derived, never stored (spec §2): the sum of every item's
    /// current (grams-scaled) nutrition.
    public var totals: NutritionTotals {
        items.reduce(NutritionTotals()) { $0 + $1.totals }
    }

    /// A reasonable default display name when none is supplied explicitly: the
    /// most-prominent (first) detected item, else the meal type/slot.
    public static func defaultName(items: [MealItem], mealType: MealType) -> String {
        if let first = items.first, !first.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return first.name.capitalized
        }
        return mealType.rawValue.capitalized
    }

    enum CodingKeys: String, CodingKey {
        case id, name, createdAt, mealType, imagePath, origin, barcodeSource, items, volumeMl, maxHeightMm
        /// Legacy-only key (pre-P2 schema): a whole-entry `FoodScanResult` carrying
        /// totals directly instead of a per-item `[MealItem]` breakdown. Decode-only —
        /// never written by `encode(to:)`.
        case legacyScanResult = "scanResult"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        mealType = try container.decode(MealType.self, forKey: .mealType)
        imagePath = try container.decodeIfPresent(String.self, forKey: .imagePath)
        barcodeSource = try container.decodeIfPresent(BarcodeSourceMetadata.self, forKey: .barcodeSource)
        origin = try container.decodeIfPresent(MealOrigin.self, forKey: .origin)
            ?? (barcodeSource == nil ? (imagePath == nil ? .manual : .photo) : .barcode)
        volumeMl = try container.decodeIfPresent(Double.self, forKey: .volumeMl)
        maxHeightMm = try container.decodeIfPresent(Double.self, forKey: .maxHeightMm)

        if let decodedItems = try container.decodeIfPresent([MealItem].self, forKey: .items), !decodedItems.isEmpty {
            items = decodedItems
            name = try container.decodeIfPresent(String.self, forKey: .name)
                ?? Self.defaultName(items: decodedItems, mealType: mealType)
            return
        }

        // Migration path: pre-P2 entries stored a whole-entry `FoodScanResult`
        // (totals only — no per-entry `items: [MealItem]` breakdown) under
        // `scanResult`. Synthesize a single item named after the entry, based at
        // the entry's own stored totals. Nominal grams come from the legacy scan
        // result's own first item if one survives (it usually does — `FoodScanResult`
        // has always carried `items: [ScanItem]`), else fall back to 100 g.
        guard let legacy = try container.decodeIfPresent(FoodScanResult.self, forKey: .legacyScanResult) else {
            throw DecodingError.dataCorruptedError(
                forKey: .items,
                in: container,
                debugDescription: "MealLogEntry requires either `items` or a legacy `scanResult`."
            )
        }

        let syntheticName = mealType.rawValue.capitalized
        let nominalGrams = legacy.items.first?.estimatedGrams ?? 100
        items = [
            MealItem(
                name: syntheticName,
                grams: nominalGrams,
                base: MealItemBase(
                    grams: nominalGrams,
                    kcal: legacy.totalCalories,
                    protein: legacy.proteinGrams,
                    fat: legacy.fatGrams,
                    carbs: legacy.carbsGrams
                )
            )
        ]
        name = syntheticName
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(name, forKey: .name)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(mealType, forKey: .mealType)
        try container.encodeIfPresent(imagePath, forKey: .imagePath)
        try container.encode(origin, forKey: .origin)
        try container.encodeIfPresent(barcodeSource, forKey: .barcodeSource)
        try container.encode(items, forKey: .items)
        try container.encodeIfPresent(volumeMl, forKey: .volumeMl)
        try container.encodeIfPresent(maxHeightMm, forKey: .maxHeightMm)
    }
}

public protocol MealLogStore: Sendable {
    func save(_ entry: MealLogEntry) async throws
    func update(_ entry: MealLogEntry) async throws
    func delete(id: UUID) async throws
    func fetchAll() async throws -> [MealLogEntry]
}

public protocol UserPreferencesStore: Sendable {
    func loadDailyTarget() async throws -> DailyNutritionTarget?
    func saveDailyTarget(_ target: DailyNutritionTarget) async throws
    func loadUserProfile() async throws -> UserProfile?
    func saveUserProfile(_ profile: UserProfile) async throws
    func loadCapturePreferences() async throws -> CapturePreferences?
    func saveCapturePreferences(_ preferences: CapturePreferences) async throws
}

public struct WeightEntry: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var date: Date
    public var weightKg: Double

    public init(id: UUID = UUID(), date: Date = Date(), weightKg: Double) {
        self.id = id
        self.date = date
        self.weightKg = weightKg
    }
}

public protocol WeightLogStore: Sendable {
    func save(_ entry: WeightEntry) async throws
    func fetchAll() async throws -> [WeightEntry]
}

public actor InMemoryMealLogStore: MealLogStore {
    private var entries: [MealLogEntry] = []

    public init() {}

    public func save(_ entry: MealLogEntry) async throws {
        entries.append(entry)
        entries.sort { $0.createdAt > $1.createdAt }
    }

    public func update(_ entry: MealLogEntry) async throws {
        guard let index = entries.firstIndex(where: { $0.id == entry.id }) else {
            return
        }
        entries[index] = entry
        entries.sort { $0.createdAt > $1.createdAt }
    }

    public func delete(id: UUID) async throws {
        entries.removeAll { $0.id == id }
    }

    public func fetchAll() async throws -> [MealLogEntry] {
        entries
    }
}

public actor InMemoryUserPreferencesStore: UserPreferencesStore {
    private var dailyTarget: DailyNutritionTarget?
    private var userProfile: UserProfile?
    private var capturePreferences: CapturePreferences?

    public init(
        initialDailyTarget: DailyNutritionTarget? = nil,
        initialUserProfile: UserProfile? = nil,
        initialCapturePreferences: CapturePreferences? = nil
    ) {
        self.dailyTarget = initialDailyTarget
        self.userProfile = initialUserProfile
        self.capturePreferences = initialCapturePreferences
    }

    public func loadDailyTarget() async throws -> DailyNutritionTarget? {
        dailyTarget
    }

    public func saveDailyTarget(_ target: DailyNutritionTarget) async throws {
        dailyTarget = target
    }

    public func loadUserProfile() async throws -> UserProfile? {
        userProfile
    }

    public func saveUserProfile(_ profile: UserProfile) async throws {
        userProfile = profile
    }

    public func loadCapturePreferences() async throws -> CapturePreferences? {
        capturePreferences
    }

    public func saveCapturePreferences(_ preferences: CapturePreferences) async throws {
        capturePreferences = preferences
    }
}

public actor InMemoryWeightLogStore: WeightLogStore {
    private var entries: [WeightEntry] = []

    public init() {}

    public func save(_ entry: WeightEntry) async throws {
        entries.append(entry)
        entries.sort { $0.date < $1.date }
    }

    public func fetchAll() async throws -> [WeightEntry] {
        entries
    }
}
