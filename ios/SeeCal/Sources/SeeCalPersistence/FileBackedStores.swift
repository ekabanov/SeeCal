import Foundation
import SeeCalDomain

/// Well-known on-disk locations for SeeCal's persisted state. All file-backed stores
/// live under `Application Support/SeeCal/` so their contents survive process death
/// and are excluded from iCloud/backup purging rules that apply to `tmp/` and caches.
public enum FileBackedStoreLocations {
    public static var applicationSupportDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("SeeCal", isDirectory: true)
    }

    public static var mealLogFileURL: URL {
        applicationSupportDirectory.appendingPathComponent("meal_log.json")
    }

    public static var preferencesFileURL: URL {
        applicationSupportDirectory.appendingPathComponent("preferences.json")
    }

    public static var weightLogFileURL: URL {
        applicationSupportDirectory.appendingPathComponent("weight_log.json")
    }

    public static var imagesDirectory: URL {
        applicationSupportDirectory.appendingPathComponent("images", isDirectory: true)
    }
}

/// Shared JSON read/write helpers used by the file-backed stores below. Writes are
/// atomic so a crash mid-write can never leave a half-written, corrupt file behind.
enum FileBackedStoreIO {
    static func read<T: Decodable>(_ type: T.Type, from fileURL: URL) -> T? {
        guard let data = try? Data(contentsOf: fileURL), !data.isEmpty else {
            return nil
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(T.self, from: data)
    }

    static func write<T: Encodable>(_ value: T, to fileURL: URL) throws {
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(value)
        try data.write(to: fileURL, options: .atomic)
    }
}

/// JSON-file backed `MealLogStore`. Loads its full contents once on init and keeps
/// them in memory as the actor's source of truth, persisting the whole collection
/// back to disk (atomically) after every mutation.
public actor FileBackedMealLogStore: MealLogStore {
    private let fileURL: URL
    private var entries: [MealLogEntry]

    public init(fileURL: URL = FileBackedStoreLocations.mealLogFileURL) {
        self.fileURL = fileURL
        self.entries = FileBackedStoreIO.read([MealLogEntry].self, from: fileURL) ?? []
    }

    public func save(_ entry: MealLogEntry) async throws {
        entries.append(entry)
        entries.sort { $0.createdAt > $1.createdAt }
        try FileBackedStoreIO.write(entries, to: fileURL)
    }

    public func update(_ entry: MealLogEntry) async throws {
        guard let index = entries.firstIndex(where: { $0.id == entry.id }) else {
            return
        }
        entries[index] = entry
        entries.sort { $0.createdAt > $1.createdAt }
        try FileBackedStoreIO.write(entries, to: fileURL)
    }

    public func delete(id: UUID) async throws {
        entries.removeAll { $0.id == id }
        try FileBackedStoreIO.write(entries, to: fileURL)
    }

    public func fetchAll() async throws -> [MealLogEntry] {
        entries
    }
}

private struct PersistedPreferences: Codable {
    var dailyTarget: DailyNutritionTarget? = nil
    var userProfile: UserProfile? = nil
}

/// JSON-file backed `UserPreferencesStore`. Both the daily target and the user
/// profile live in a single small JSON document.
public actor FileBackedUserPreferencesStore: UserPreferencesStore {
    private let fileURL: URL
    private var payload: PersistedPreferences

    public init(fileURL: URL = FileBackedStoreLocations.preferencesFileURL) {
        self.fileURL = fileURL
        self.payload = FileBackedStoreIO.read(PersistedPreferences.self, from: fileURL) ?? PersistedPreferences()
    }

    public func loadDailyTarget() async throws -> DailyNutritionTarget? {
        payload.dailyTarget
    }

    public func saveDailyTarget(_ target: DailyNutritionTarget) async throws {
        payload.dailyTarget = target
        try FileBackedStoreIO.write(payload, to: fileURL)
    }

    public func loadUserProfile() async throws -> UserProfile? {
        payload.userProfile
    }

    public func saveUserProfile(_ profile: UserProfile) async throws {
        payload.userProfile = profile
        try FileBackedStoreIO.write(payload, to: fileURL)
    }
}

/// JSON-file backed `WeightLogStore`.
public actor FileBackedWeightLogStore: WeightLogStore {
    private let fileURL: URL
    private var entries: [WeightEntry]

    public init(fileURL: URL = FileBackedStoreLocations.weightLogFileURL) {
        self.fileURL = fileURL
        self.entries = FileBackedStoreIO.read([WeightEntry].self, from: fileURL) ?? []
    }

    public func save(_ entry: WeightEntry) async throws {
        entries.append(entry)
        entries.sort { $0.date < $1.date }
        try FileBackedStoreIO.write(entries, to: fileURL)
    }

    public func fetchAll() async throws -> [WeightEntry] {
        entries
    }
}
