import Foundation

/// Settings §8 user-controllable capture behavior, persisted independently of
/// the profile/goal.
public struct CapturePreferences: Codable, Equatable, Sendable {
    /// "Capture coaching — Distance & level guides". Gates the camera screen's
    /// level indicator and hint text; the shutter itself is never gated by
    /// this (spec §5: "coaching, never gating — shutter always enabled").
    public var captureCoachingEnabled: Bool

    public init(captureCoachingEnabled: Bool = true) {
        self.captureCoachingEnabled = captureCoachingEnabled
    }

    public static let `default` = CapturePreferences()
}
