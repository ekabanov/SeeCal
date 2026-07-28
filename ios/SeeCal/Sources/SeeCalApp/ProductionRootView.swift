import SwiftUI
import SeeCalInference
import os
#if canImport(UIKit)
import UIKit
#endif

public struct ProductionRootView: View {
    private let logger = Logger(subsystem: "SeeCal", category: "ProductionRootView")
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
            logger.log("Memory warning — releasing MLX caches.")
            SeeCalMemory.releaseCaches()
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
        logger.log("Simulator build — using development mock engine (MLX-Metal unavailable on simulator).")
        print("[SeeCal][ProductionRootView] simulator: using development mock engine")
        loadState = .loaded(SeeCalBootstrap.makeDevelopmentViewModel())
#else
        do {
            logger.log("Starting model load for model id: \(config.modelPath, privacy: .public)")
            print("[SeeCal][ProductionRootView] load start modelPath=\(config.modelPath)")
            let vm = try await SeeCalBootstrap.makeProductionViewModelUsingMLX(config: config)
            loadState = .loaded(vm)
            logger.log("Model load succeeded for model id: \(config.modelPath, privacy: .public)")
            print("[SeeCal][ProductionRootView] load success")
        } catch {
            let message = debugMessage(for: error)
            logger.error("Model load failed for model id: \(config.modelPath, privacy: .public). Error: \(message, privacy: .public)")
            print("[SeeCal][ProductionRootView] load failed error=\(message)")
            loadState = .failed(message)
        }
#endif
    }

    private func debugMessage(for error: Error) -> String {
        String(describing: error)
    }
}
