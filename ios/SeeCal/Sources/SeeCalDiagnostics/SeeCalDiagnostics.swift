import Foundation
import OSLog

public enum DiagnosticLevel: String, Codable, Sendable {
    case debug
    case info
    case notice
    case error
    case fault

    fileprivate var osLogType: OSLogType {
        switch self {
        case .debug: return .debug
        case .info: return .info
        case .notice: return .default
        case .error: return .error
        case .fault: return .fault
        }
    }
}

public struct DiagnosticEvent: Codable, Equatable, Sendable {
    public let timestamp: Date
    public let sessionID: UUID
    public let level: DiagnosticLevel
    public let category: String
    public let name: String
    public let fields: [String: String]

    public init(
        timestamp: Date,
        sessionID: UUID,
        level: DiagnosticLevel,
        category: String,
        name: String,
        fields: [String: String]
    ) {
        self.timestamp = timestamp
        self.sessionID = sessionID
        self.level = level
        self.category = category
        self.name = name
        self.fields = fields
    }
}

public struct DiagnosticReportMetadata: Equatable, Sendable {
    public var appVersion: String
    public var buildNumber: String
    public var operatingSystem: String
    public var deviceModel: String
    public var modelLabel: String
    public var adapterVersion: String
    public var quantization: String

    public init(
        appVersion: String,
        buildNumber: String,
        operatingSystem: String,
        deviceModel: String,
        modelLabel: String,
        adapterVersion: String,
        quantization: String
    ) {
        self.appVersion = appVersion
        self.buildNumber = buildNumber
        self.operatingSystem = operatingSystem
        self.deviceModel = deviceModel
        self.modelLabel = modelLabel
        self.adapterVersion = adapterVersion
        self.quantization = quantization
    }
}

public struct DiagnosticLogConfiguration: Equatable, Sendable {
    public var directory: URL
    public var maximumFileBytes: Int
    public var retainedFileCount: Int
    public var retentionInterval: TimeInterval

    public init(
        directory: URL,
        maximumFileBytes: Int = 1_048_576,
        retainedFileCount: Int = 4,
        retentionInterval: TimeInterval = 7 * 24 * 60 * 60
    ) {
        self.directory = directory
        self.maximumFileBytes = max(4_096, maximumFileBytes)
        self.retainedFileCount = max(1, retainedFileCount)
        self.retentionInterval = max(60, retentionInterval)
    }

    public static var appDefault: Self {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return Self(directory: base.appendingPathComponent("SeeCal/Diagnostics", isDirectory: true))
    }
}

public enum DiagnosticReportError: Error, Equatable {
    case couldNotCreateReport
}

/// Synchronous, lock-isolated log storage. Every line is flushed before `record`
/// returns so the latest breadcrumb survives a crash or jetsam termination.
///
/// Callers pass only deliberately selected technical fields. Values still pass
/// through a final path/control-character scrub before they reach disk.
public final class DiagnosticLogStore: @unchecked Sendable {
    private let configuration: DiagnosticLogConfiguration
    private let fileManager: FileManager
    private let now: @Sendable () -> Date
    private let sessionID: UUID
    private let lock = NSLock()
    private let encoder: JSONEncoder
    private var didPrepareDirectory = false

    public init(
        configuration: DiagnosticLogConfiguration,
        fileManager: FileManager = .default,
        sessionID: UUID = UUID(),
        now: @escaping @Sendable () -> Date = Date.init
    ) {
        self.configuration = configuration
        self.fileManager = fileManager
        self.sessionID = sessionID
        self.now = now
        self.encoder = JSONEncoder()
        self.encoder.dateEncodingStrategy = .iso8601
        self.encoder.outputFormatting = [.sortedKeys]
    }

    public func record(
        _ level: DiagnosticLevel,
        category: String,
        name: String,
        fields: [String: String] = [:]
    ) {
        lock.lock()
        defer { lock.unlock() }

        do {
            try prepareDirectoryIfNeeded()
            let event = DiagnosticEvent(
                timestamp: now(),
                sessionID: sessionID,
                level: level,
                category: Self.safeToken(category),
                name: Self.safeToken(name),
                fields: Self.sanitized(fields)
            )
            var line = try encoder.encode(event)
            line.append(0x0A)
            if try activeLogSize() + line.count > configuration.maximumFileBytes {
                try rotateFiles()
            }
            try appendAndSynchronize(line)
        } catch {
            // Diagnostics must never change product behavior. The mirrored OSLog
            // entry below still gives attached-device sessions a breadcrumb.
        }

        let logger = Logger(subsystem: "SeeCal", category: Self.safeToken(category))
        let renderedFields = Self.rendered(Self.sanitized(fields))
        logger.log(level: level.osLogType, "\(Self.safeToken(name), privacy: .public) \(renderedFields, privacy: .public)")
    }

    public func exportReport(metadata: DiagnosticReportMetadata) throws -> URL {
        lock.lock()
        defer { lock.unlock() }

        try prepareDirectoryIfNeeded()
        let logData = try combinedLogData()
        let generatedAt = now()
        let timestamp = Self.fileTimestamp(generatedAt)
        let reportURL = fileManager.temporaryDirectory
            .appendingPathComponent("seecal-diagnostics-\(timestamp)-\(UUID().uuidString.prefix(8)).txt")

        var report = """
        SeeCal Diagnostics
        Generated: \(Self.displayTimestamp(generatedAt))

        PRIVACY
        This report contains technical events, app/device/model versions, timings,
        memory measurements, and error codes. It does not contain meal photos,
        profile values, meal or ingredient names, nutrition values, prompts, raw
        model output, or absolute file paths.

        ENVIRONMENT
        App version: \(Self.singleLine(metadata.appVersion))
        Build: \(Self.singleLine(metadata.buildNumber))
        Operating system: \(Self.singleLine(metadata.operatingSystem))
        Device model: \(Self.singleLine(metadata.deviceModel))
        Model: \(Self.singleLine(metadata.modelLabel))
        Adapter: \(Self.singleLine(metadata.adapterVersion))
        Quantization: \(Self.singleLine(metadata.quantization))

        EVENTS (oldest first, JSON Lines)

        """
        if logData.isEmpty {
            report += "(no diagnostic events were available)\n"
        } else if let logs = String(data: logData, encoding: .utf8) {
            report += logs
            if !report.hasSuffix("\n") { report += "\n" }
        }

        guard let data = report.data(using: .utf8) else {
            throw DiagnosticReportError.couldNotCreateReport
        }
        try data.write(to: reportURL, options: [.atomic])
        return reportURL
    }

    public func recordedEvents() throws -> [DiagnosticEvent] {
        lock.lock()
        defer { lock.unlock() }
        try prepareDirectoryIfNeeded()
        return try Self.decodeEvents(from: combinedLogData())
    }

    private var activeLogURL: URL {
        configuration.directory.appendingPathComponent("current.log")
    }

    private func rotatedLogURL(_ index: Int) -> URL {
        configuration.directory.appendingPathComponent("previous-\(index).log")
    }

    private func prepareDirectoryIfNeeded() throws {
        guard !didPrepareDirectory else { return }
        try fileManager.createDirectory(at: configuration.directory, withIntermediateDirectories: true)
        try pruneExpiredFiles()
        if fileManager.fileExists(atPath: activeLogURL.path) {
            try rotateFiles()
        }
        didPrepareDirectory = true
    }

    private func pruneExpiredFiles() throws {
        let cutoff = now().addingTimeInterval(-configuration.retentionInterval)
        let urls = try fileManager.contentsOfDirectory(
            at: configuration.directory,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )
        for url in urls where url.pathExtension == "log" {
            let values = try? url.resourceValues(forKeys: [.contentModificationDateKey])
            if let modified = values?.contentModificationDate, modified < cutoff {
                try? fileManager.removeItem(at: url)
            }
        }
    }

    private func activeLogSize() throws -> Int {
        guard fileManager.fileExists(atPath: activeLogURL.path) else { return 0 }
        let attributes = try fileManager.attributesOfItem(atPath: activeLogURL.path)
        return (attributes[.size] as? NSNumber)?.intValue ?? 0
    }

    private func appendAndSynchronize(_ data: Data) throws {
        if !fileManager.fileExists(atPath: activeLogURL.path) {
            _ = fileManager.createFile(atPath: activeLogURL.path, contents: nil)
            #if os(iOS)
            try? fileManager.setAttributes(
                [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication],
                ofItemAtPath: activeLogURL.path
            )
            #endif
        }
        let handle = try FileHandle(forWritingTo: activeLogURL)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: data)
        try handle.synchronize()
    }

    private func rotateFiles() throws {
        let oldest = rotatedLogURL(configuration.retainedFileCount)
        try? fileManager.removeItem(at: oldest)
        if configuration.retainedFileCount > 1 {
            for index in stride(from: configuration.retainedFileCount - 1, through: 1, by: -1) {
                let source = rotatedLogURL(index)
                guard fileManager.fileExists(atPath: source.path) else { continue }
                try? fileManager.moveItem(at: source, to: rotatedLogURL(index + 1))
            }
        }
        if fileManager.fileExists(atPath: activeLogURL.path) {
            try? fileManager.moveItem(at: activeLogURL, to: rotatedLogURL(1))
        }
    }

    private func combinedLogData() throws -> Data {
        var result = Data()
        for index in stride(from: configuration.retainedFileCount, through: 1, by: -1) {
            let url = rotatedLogURL(index)
            if let data = try? Data(contentsOf: url) {
                result.append(data)
            }
        }
        if let data = try? Data(contentsOf: activeLogURL) {
            result.append(data)
        }
        return result
    }

    private static func decodeEvents(from data: Data) throws -> [DiagnosticEvent] {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try data.split(separator: 0x0A).map {
            try decoder.decode(DiagnosticEvent.self, from: Data($0))
        }
    }

    private static func sanitized(_ fields: [String: String]) -> [String: String] {
        Dictionary(uniqueKeysWithValues: fields.map { key, value in
            (safeToken(key), sanitizedValue(value))
        })
    }

    private static func safeToken(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        let scalars = value.unicodeScalars.map { allowed.contains($0) ? Character(String($0)) : "_" }
        return String(scalars.prefix(64))
    }

    private static func sanitizedValue(_ value: String) -> String {
        let singleLine = singleLine(value)
        let pathMarkers = ["/private/", "/var/", "/Users/", "file://"]
        guard !pathMarkers.contains(where: singleLine.contains) else {
            return "<redacted-path>"
        }
        return String(singleLine.prefix(256))
    }

    private static func singleLine(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\t", with: " ")
    }

    private static func rendered(_ fields: [String: String]) -> String {
        fields.keys.sorted().map { "\($0)=\(fields[$0] ?? "")" }.joined(separator: " ")
    }

    private static func displayTimestamp(_ date: Date) -> String {
        ISO8601DateFormatter().string(from: date)
    }

    private static func fileTimestamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: date)
    }
}

public enum SeeCalDiagnostics {
    private static let store = DiagnosticLogStore(configuration: .appDefault)

    public static func record(
        _ level: DiagnosticLevel = .info,
        category: String,
        name: String,
        fields: [String: String] = [:]
    ) {
        store.record(level, category: category, name: name, fields: fields)
    }

    public static func errorFields(_ error: Error) -> [String: String] {
        let nsError = error as NSError
        return [
            "error_type": String(reflecting: type(of: error)),
            "error_domain": nsError.domain,
            "error_code": String(nsError.code)
        ]
    }

    public static func exportReport(metadata: DiagnosticReportMetadata) throws -> URL {
        try store.exportReport(metadata: metadata)
    }
}
