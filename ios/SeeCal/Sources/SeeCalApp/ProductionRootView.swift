import SwiftUI
import SeeCalDiagnostics
import SeeCalInference
#if canImport(UIKit)
import UIKit
#endif

public struct ProductionRootView: View {
    public enum LoadState {
        case loading
        case loaded(AppViewModel)
        case failed(String)
    }

    private let config: QwenRuntimeConfig
    @State private var loadState: LoadState = .loading

    public init(config: QwenRuntimeConfig) {
        self.config = config
    }

    public var body: some View {
        Group {
            switch loadState {
            case .loading:
                StartupLoadingView()
            case let .loaded(viewModel):
                RootView(viewModel: viewModel)
            case let .failed(message):
                StartupFailureView(message: message) {
                    loadState = .loading
                    Task {
                        await load()
                    }
                }
            }
        }
        .task {
            if case .loading = loadState {
                await load()
            }
        }
#if canImport(UIKit)
        // Shed MLX's cached GPU buffers when the OS warns about memory, so a long
        // session creeping toward the jetsam limit recovers instead of being
        // killed. Cheap and safe — the cache repopulates on the next inference.
        .onReceive(NotificationCenter.default.publisher(
            for: UIApplication.didReceiveMemoryWarningNotification
        )) { _ in
            SeeCalDiagnostics.record(
                .fault,
                category: "memory",
                name: "memory_warning_received"
            )
            SeeCalMemory.releaseCaches()
            SeeCalDiagnostics.record(
                .notice,
                category: "memory",
                name: "mlx_cache_released_after_warning"
            )
        }
#endif
    }

    @MainActor
    private func load() async {
#if targetEnvironment(simulator)
        // MLX-Metal cannot create a GPU device on the iOS Simulator — it aborts
        // the process (SIGABRT in mlx::core::metal::Device), which is a C++
        // abort no Swift `catch` can intercept. Simulator builds therefore run
        // the mock inference engine so the full UI is usable for development and
        // visual QA. Device builds always use MLX (the #else branch below).
        SeeCalDiagnostics.record(
            .notice,
            category: "model_load",
            name: "simulator_mock_engine_selected"
        )
        loadState = .loaded(SeeCalBootstrap.makeDevelopmentViewModel())
#else
        do {
            let startedAt = DispatchTime.now().uptimeNanoseconds
            SeeCalDiagnostics.record(
                .notice,
                category: "model_load",
                name: "production_model_load_started",
                fields: ["adapter_configured": String(config.adapterPath != nil)]
            )
            let vm = try await SeeCalBootstrap.makeProductionViewModelUsingMLX(config: config)
            loadState = .loaded(vm)
            SeeCalDiagnostics.record(
                .notice,
                category: "model_load",
                name: "production_model_load_succeeded",
                fields: [
                    "duration_ms": String((DispatchTime.now().uptimeNanoseconds - startedAt) / 1_000_000)
                ]
            )
        } catch {
            let message = debugMessage(for: error)
            SeeCalDiagnostics.record(
                .fault,
                category: "model_load",
                name: "production_model_load_failed",
                fields: SeeCalDiagnostics.errorFields(error)
            )
            loadState = .failed(message)
        }
#endif
    }

    private func debugMessage(for error: Error) -> String {
        String(describing: error)
    }
}
