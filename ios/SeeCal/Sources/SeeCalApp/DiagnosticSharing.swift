import Foundation
import SeeCalDiagnostics
import SwiftUI

#if os(iOS)
import MessageUI
import UIKit
#endif

enum DiagnosticReportFactory {
    static func makeReport(for viewModel: AppViewModel) throws -> URL {
        try SeeCalDiagnostics.exportReport(metadata: metadata(for: viewModel))
    }

    static func metadata(for viewModel: AppViewModel) -> DiagnosticReportMetadata {
        let info = Bundle.main.infoDictionary ?? [:]
        let version = info["CFBundleShortVersionString"] as? String ?? "unknown"
        let build = info["CFBundleVersion"] as? String ?? "unknown"

        return DiagnosticReportMetadata(
            appVersion: version,
            buildNumber: build,
            operatingSystem: operatingSystem,
            deviceModel: deviceModel,
            modelLabel: viewModel.modelInfo.modelLabel,
            adapterVersion: viewModel.modelInfo.adapterVersionLabel ?? "none",
            quantization: viewModel.modelInfo.quantizationLabel ?? "unknown"
        )
    }

    static var configuredRecipient: String? {
        guard
            let value = Bundle.main.object(forInfoDictionaryKey: "SeeCalDiagnosticsEmail") as? String
        else {
            return nil
        }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.contains("@") ? trimmed : nil
    }

    static var canSendMail: Bool {
        #if os(iOS)
        MFMailComposeViewController.canSendMail()
        #else
        false
        #endif
    }

    private static var operatingSystem: String {
        #if os(iOS)
        "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)"
        #else
        ProcessInfo.processInfo.operatingSystemVersionString
        #endif
    }

    private static var deviceModel: String {
        if let simulator = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] {
            return "\(simulator) (Simulator)"
        }
        var systemInfo = utsname()
        guard uname(&systemInfo) == 0 else {
            #if os(iOS)
            return UIDevice.current.model
            #else
            return "unknown"
            #endif
        }
        let capacity = MemoryLayout.size(ofValue: systemInfo.machine)
        return withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: capacity) {
                String(cString: $0)
            }
        }
    }
}

#if os(iOS)
struct DiagnosticMailComposeView: UIViewControllerRepresentable {
    let reportURL: URL
    let recipient: String?

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIViewController(context: Context) -> MFMailComposeViewController {
        let controller = MFMailComposeViewController()
        controller.mailComposeDelegate = context.coordinator
        if let recipient {
            controller.setToRecipients([recipient])
        }
        controller.setSubject("SeeCal diagnostic report")
        controller.setMessageBody(
            """
            Please describe what you were doing, what you expected, and what happened.

            The attached report contains technical diagnostics. It excludes meal photos, profile values, meal and ingredient names, nutrition values, prompts, and raw model output.
            """,
            isHTML: false
        )
        if let data = try? Data(contentsOf: reportURL) {
            controller.addAttachmentData(data, mimeType: "text/plain", fileName: reportURL.lastPathComponent)
        }
        return controller
    }

    func updateUIViewController(_ uiViewController: MFMailComposeViewController, context: Context) {}

    final class Coordinator: NSObject, MFMailComposeViewControllerDelegate {
        func mailComposeController(
            _ controller: MFMailComposeViewController,
            didFinishWith result: MFMailComposeResult,
            error: Error?
        ) {
            let resultName: String
            switch result {
            case .cancelled: resultName = "cancelled"
            case .saved: resultName = "saved"
            case .sent: resultName = "sent"
            case .failed: resultName = "failed"
            @unknown default: resultName = "unknown"
            }
            var fields = ["result": resultName]
            if let error {
                fields.merge(SeeCalDiagnostics.errorFields(error)) { current, _ in current }
            }
            SeeCalDiagnostics.record(
                result == .failed ? .error : .notice,
                category: "diagnostics",
                name: "mail_compose_finished",
                fields: fields
            )
            controller.dismiss(animated: true)
        }
    }
}

struct DiagnosticActivityView: UIViewControllerRepresentable {
    let reportURL: URL

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(
            activityItems: [
                "SeeCal diagnostic report. The attached file excludes meal photos and personal nutrition/profile data.",
                reportURL
            ],
            applicationActivities: nil
        )
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
#endif
