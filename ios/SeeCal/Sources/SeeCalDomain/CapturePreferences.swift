import Foundation

/// Settings §8 "Capture" card toggles — user-controllable capture behavior,
/// persisted independently of the profile/goal. Both default to `true`,
/// matching the prototype's `.switch[aria-checked="true"]` starting state for
/// both rows.
public struct CapturePreferences: Codable, Equatable, Sendable {
    /// "Use LiDAR depth — Improves portion-size accuracy". Read by the
    /// (currently dormant) depth-capture path once it lands (depth-plan D5);
    /// today it additionally gates the camera screen's depth affordances
    /// alongside `CaptureService.supportsDepthCapture`, which is false on every
    /// device until D5 ships.
    public var useLiDARDepth: Bool
    /// "Capture coaching — Distance & level guides". Gates the camera screen's
    /// level indicator and hint text; the shutter itself is never gated by
    /// this (spec §5: "coaching, never gating — shutter always enabled").
    public var captureCoachingEnabled: Bool

    public init(useLiDARDepth: Bool = true, captureCoachingEnabled: Bool = true) {
        self.useLiDARDepth = useLiDARDepth
        self.captureCoachingEnabled = captureCoachingEnabled
    }

    public static let `default` = CapturePreferences()
}
