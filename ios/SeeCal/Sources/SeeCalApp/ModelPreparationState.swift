import Combine
import SeeCalInference

/// UI-facing projection of the lazy MLX engine lifecycle. Production starts at
/// `.notStarted`; development/test view models default to `.ready` because
/// their injected runners do not have a model-loading phase.
@MainActor
public final class ModelPreparationState: ObservableObject {
    public enum Phase: Equatable {
        case notStarted
        case loading
        case ready
    }

    @Published public private(set) var phase: Phase

    public init(phase: Phase = .ready) {
        self.phase = phase
    }

    public func update(from loadState: MLXModelLoadState) {
        switch loadState {
        case .notStarted:
            phase = .notStarted
        case .loading:
            phase = .loading
        case .ready:
            phase = .ready
        }
    }
}
