import SwiftUI
import SeeCalDomain

// MARK: - Weekly-rate display formatting

/// Exact label/kind/note strings from the prototype's `renderProfile()`
/// (`docs/design/prototype/seecal-prototype.html` ~line 1474), spec §2/§3.
/// All signs use the typographic minus (U+2212), matching the prototype copy.
public enum WeeklyRateFormat {
    /// "−0.5 kg / week" / "+0.3 kg / week"; "0 — maintain" at (effectively) zero.
    /// Prototype: `(r>0?"+":"−")+Math.abs(r).toFixed(1)+" kg / week"`, replaced
    /// by `"0 — maintain"` when `|r| < 0.049`.
    public static func label(for rate: Double) -> String {
        if abs(rate) < 0.049 { return "0 — maintain" }
        let sign = rate > 0 ? "+" : "−"
        return sign + String(format: "%.1f", abs(rate)) + " kg / week"
    }

    /// "Lose weight" / "Maintain" / "Gain weight" (rendered uppercase by the
    /// component, matching `.rateKind{text-transform:uppercase}`).
    public static func kind(for rate: Double) -> String {
        if rate < -0.049 { return "Lose weight" }
        if rate > 0.049 { return "Gain weight" }
        return "Maintain"
    }

    /// Warn styling applies only outside the recommended band (spec §2:
    /// below −0.75 or above +0.25).
    public static func isWarning(for rate: Double) -> Bool {
        switch GoalCalculator.weeklyRateBand(for: rate) {
        case .aggressiveLoss, .aggressiveGain:
            return true
        case .recommended, .neutral:
            return false
        }
    }

    /// Note line under the slider — the two exact warning strings from spec §2,
    /// or the standing recommendation when inside the band.
    public static func note(for rate: Double) -> String {
        switch GoalCalculator.weeklyRateBand(for: rate) {
        case .aggressiveLoss:
            return "Faster than 0.75 kg/week is hard to sustain and not recommended."
        case .aggressiveGain:
            return "Gaining faster than 0.25 kg/week mostly adds fat."
        case .recommended, .neutral:
            return "Recommended: lose 0.25–0.75 kg per week."
        }
    }
}

// MARK: - Weekly-rate slider

/// Prototype `.ratebox`: header row (live value label + kind tag), track with the
/// recommended band highlighted behind a native slider, endpoint scale labels,
/// and a note line with a warn state. Range −1.0…+0.5, step 0.1 (spec §2).
///
/// The band rectangle sits at 16.7%…50% of the track width — the −0.75…−0.25
/// recommended band mapped onto the −1.0…+0.5 range, exactly as `.rateband`
/// (`left:16.7%; width:33.3%`) positions it in the prototype CSS.
public struct WeeklyRateSlider: View {
    @Binding private var value: Double

    public init(value: Binding<Double>) {
        _value = value
    }

    /// Snaps writes to the 0.1 step and clamps to the spec range, so float noise
    /// from the native slider never reaches the bound model value.
    private var snappedValue: Binding<Double> {
        Binding(
            get: { value },
            set: { value = UserProfile.clampWeeklyRate(($0 * 10).rounded() / 10) }
        )
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // `.ratehead` — live label + kind tag.
            HStack(alignment: .firstTextBaseline) {
                Text(WeeklyRateFormat.label(for: value))
                    .font(.system(size: 17, weight: .heavy))
                    .monospacedDigit()
                    .foregroundStyle(Theme.appInk)
                Spacer()
                Text(WeeklyRateFormat.kind(for: value).uppercased())
                    .font(.system(size: 12, weight: .bold))
                    .tracking(0.6) // .05em of 12px
                    .foregroundStyle(Theme.basil)
            }
            .padding(.bottom, 10)

            // `.ratewrap` — recommended band behind a native slider.
            ZStack {
                GeometryReader { geometry in
                    // left:16.7%, width:33.3% — center at 1/3 of the track.
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .fill(Theme.basilSoft)
                        .frame(width: geometry.size.width / 3, height: 14)
                        .position(x: geometry.size.width / 3, y: geometry.size.height / 2)
                }
                Slider(value: snappedValue, in: UserProfile.weeklyRateRange, step: 0.1)
                    .tint(Theme.basil)
                    .accessibilityLabel("Target weight change per week")
                    .accessibilityValue(WeeklyRateFormat.label(for: value))
            }
            .frame(height: 28)

            // `.ratescale` — endpoint labels.
            HStack {
                Text("−1.0 kg")
                Spacer()
                Text("0")
                Spacer()
                Text("+0.5 kg")
            }
            .font(.system(size: 10.5, weight: .semibold))
            .monospacedDigit()
            .foregroundStyle(Theme.appInk2)
            .padding(.top, 4)

            // `.ratenote` (+ `.warn`).
            Text(WeeklyRateFormat.note(for: value))
                .font(.system(size: 11.5, weight: WeeklyRateFormat.isWarning(for: value) ? .semibold : .regular))
                .foregroundStyle(WeeklyRateFormat.isWarning(for: value) ? Theme.danger : Theme.appInk2)
                .lineSpacing(3)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 8)
        }
        .padding(.top, 6)
        .padding(.bottom, 2)
    }
}
