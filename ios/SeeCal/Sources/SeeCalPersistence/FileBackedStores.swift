import Foundation
import SeeCalDiagnostics
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
            SeeCalDiagnostics.record(
                .debug,
                category: "persistence",
                name: "store_empty_or_missing",
                fields: ["store": storeName(fileURL)]
            )
            return nil
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        do {
            let value = try decoder.decode(T.self, from: data)
            SeeCalDiagnostics.record(
                .info,
                category: "persistence",
                name: "store_read_succeeded",
                fields: ["store": storeName(fileURL), "bytes": String(data.count)]
            )
            return value
        } catch {
            // Never silently discard an unreadable store file: the next save() would
            // overwrite it, permanently destroying data a future app version might have
            // been able to migrate. Preserve the original bytes as a timestamped backup
            // and start fresh.
            SeeCalDiagnostics.record(
                .error,
                category: "persistence",
                name: "store_decode_failed",
                fields: ["store": storeName(fileURL)]
                    .merging(SeeCalDiagnostics.errorFields(error)) { current, _ in current }
            )
            backUpUnreadableFile(at: fileURL)
            return nil
        }
    }

    /// Renames an unreadable store file to `<name>.bak-<timestamp>` next to the
    /// original, keeping its bytes for later recovery. The store then proceeds as if
    /// no file existed, so subsequent saves write fresh state without clobbering the
    /// backup.
    private static func backUpUnreadableFile(at fileURL: URL) {
        let timestamp = Int(Date().timeIntervalSince1970 * 1000)
        let backupURL = fileURL.deletingLastPathComponent()
            .appendingPathComponent("\(fileURL.lastPathComponent).bak-\(timestamp)")
        do {
            try FileManager.default.moveItem(at: fileURL, to: backupURL)
            SeeCalDiagnostics.record(
                .notice,
                category: "persistence",
                name: "unreadable_store_preserved",
                fields: ["store": storeName(fileURL)]
            )
        } catch {
            SeeCalDiagnostics.record(
                .error,
                category: "persistence",
                name: "unreadable_store_backup_failed",
                fields: ["store": storeName(fileURL)]
                    .merging(SeeCalDiagnostics.errorFields(error)) { current, _ in current }
            )
        }
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
        SeeCalDiagnostics.record(
            .info,
            category: "persistence",
            name: "store_write_succeeded",
            fields: ["store": storeName(fileURL), "bytes": String(data.count)]
        )
    }

    private static func storeName(_ fileURL: URL) -> String {
        switch fileURL.lastPathComponent {
        case "meal_log.json": return "meal_log"
        case "preferences.json": return "preferences"
        case "weight_log.json": return "weight_log"
        default: return "custom_store"
        }
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
    var capturePreferences: CapturePreferences? = nil
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

    public func loadCapturePreferences() async throws -> CapturePreferences? {
        payload.capturePreferences
    }

    public func saveCapturePreferences(_ preferences: CapturePreferences) async throws {
        payload.capturePreferences = preferences
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
