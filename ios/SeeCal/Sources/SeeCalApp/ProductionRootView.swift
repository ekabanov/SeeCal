import SwiftUI
import SeeCalInference
import os

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
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Loading MLX model…")
                }
            case let .loaded(viewModel):
                RootView(viewModel: viewModel)
            case let .failed(message):
                VStack(spacing: 12) {
                    Text("Failed to load model")
                        .font(.headline)
                    Text(message)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button("Retry") {
                        Task { await load() }
                    }
                }
                .padding()
            }
        }
        .task {
            if case .loading = loadState {
                await load()
            }
        }
    }

    @MainActor
    private func load() async {
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
    }

    private func debugMessage(for error: Error) -> String {
        String(describing: error)
    }
}
