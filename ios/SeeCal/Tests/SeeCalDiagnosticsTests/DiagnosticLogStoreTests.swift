import Foundation
import XCTest
@testable import SeeCalDiagnostics

final class DiagnosticLogStoreTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("SeeCalDiagnosticsTests-\(UUID().uuidString)", isDirectory: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testRecordsStructuredEventsAndSanitizesPaths() throws {
        let sessionID = UUID()
        let store = makeStore(sessionID: sessionID)

        store.record(.notice, category: "camera", name: "capture_started", fields: [
            "scan_id": "scan-1",
            "path": "/private/var/mobile/meal.jpg",
            "multiline": "one\ntwo"
        ])

        let events = try store.recordedEvents()
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].sessionID, sessionID)
        XCTAssertEqual(events[0].category, "camera")
        XCTAssertEqual(events[0].name, "capture_started")
        XCTAssertEqual(events[0].fields["path"], "<redacted-path>")
        XCTAssertEqual(events[0].fields["multiline"], "one two")
    }

    func testRotatesFilesAndExportsOldestFirst() throws {
        let store = makeStore(maximumFileBytes: 4_096, retainedFileCount: 2)

        for index in 0..<80 {
            store.record(.info, category: "inference", name: "event", fields: [
                "index": String(index),
                "padding": String(repeating: "x", count: 120)
            ])
        }

        let events = try store.recordedEvents()
        XCTAssertFalse(events.isEmpty)
        let indices = events.compactMap { Int($0.fields["index"] ?? "") }
        XCTAssertEqual(indices, indices.sorted())
        XCTAssertEqual(indices.last, 79)

        let report = try store.exportReport(metadata: sampleMetadata)
        let text = try String(contentsOf: report, encoding: .utf8)
        XCTAssertTrue(text.contains("SeeCal Diagnostics"))
        XCTAssertTrue(text.contains("\"index\":\"79\""))
        XCTAssertTrue(text.contains("It does not contain meal photos"))
        try? FileManager.default.removeItem(at: report)
    }

    func testNewStorePreservesPreviousSession() throws {
        let first = makeStore(sessionID: UUID())
        first.record(.info, category: "app", name: "first_session")

        let second = makeStore(sessionID: UUID())
        second.record(.info, category: "app", name: "second_session")

        XCTAssertEqual(
            try second.recordedEvents().map(\.name),
            ["first_session", "second_session"]
        )
    }

    func testReportContainsEnvironmentWithoutUserContent() throws {
        let store = makeStore()
        store.record(.error, category: "persistence", name: "write_failed", fields: [
            "error_domain": "NSCocoaErrorDomain",
            "error_code": "512"
        ])

        let report = try store.exportReport(metadata: sampleMetadata)
        let text = try String(contentsOf: report, encoding: .utf8)
        XCTAssertTrue(text.contains("App version: 0.1.0"))
        XCTAssertTrue(text.contains("Adapter: v7b"))
        XCTAssertTrue(text.contains("write_failed"))
        XCTAssertFalse(text.contains("meal_log.json"))
        try? FileManager.default.removeItem(at: report)
    }

    func testErrorFieldsExcludeLocalizedDescriptionAndPaths() {
        let error = NSError(
            domain: "SeeCal.Test",
            code: 42,
            userInfo: [
                NSLocalizedDescriptionKey: "Meal data failed at /private/var/mobile/meal-log.json"
            ]
        )

        XCTAssertEqual(
            SeeCalDiagnostics.errorFields(error),
            [
                "error_domain": "SeeCal.Test",
                "error_code": "42",
                "error_type": "NSError"
            ]
        )
    }

    private func makeStore(
        maximumFileBytes: Int = 1_048_576,
        retainedFileCount: Int = 4,
        sessionID: UUID = UUID()
    ) -> DiagnosticLogStore {
        DiagnosticLogStore(
            configuration: DiagnosticLogConfiguration(
                directory: directory,
                maximumFileBytes: maximumFileBytes,
                retainedFileCount: retainedFileCount
            ),
            sessionID: sessionID
        )
    }

    private var sampleMetadata: DiagnosticReportMetadata {
        DiagnosticReportMetadata(
            appVersion: "0.1.0",
            buildNumber: "1",
            operatingSystem: "iOS 18",
            deviceModel: "iPhone",
            modelLabel: "Qwen3.5-4B",
            adapterVersion: "v7b",
            quantization: "4-bit"
        )
    }
}
