import Foundation
import SeeCalDomain

public struct MealLogEntry: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var createdAt: Date
    public var mealType: MealType
    public var imagePath: String
    public var scanResult: FoodScanResult

    public init(
        id: UUID = UUID(),
        createdAt: Date = Date(),
        mealType: MealType,
        imagePath: String,
        scanResult: FoodScanResult
    ) {
        self.id = id
        self.createdAt = createdAt
        self.mealType = mealType
        self.imagePath = imagePath
        self.scanResult = scanResult
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

    public init(initialDailyTarget: DailyNutritionTarget? = nil, initialUserProfile: UserProfile? = nil) {
        self.dailyTarget = initialDailyTarget
        self.userProfile = initialUserProfile
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
