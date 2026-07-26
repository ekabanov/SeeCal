import SwiftUI

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - Cross-platform dynamic (light/dark) Color

// `Theme` (below) needs colors that flip automatically with the system
// appearance without every call site having to read `@Environment(\.colorScheme)`.
// SwiftUI itself doesn't expose a `Color(light:dark:)` initializer, but both
// UIKit and AppKit expose an appearance-driven dynamic color constructor — this
// wraps whichever is available so the same `Theme` code compiles for the iOS app
// target and for the macOS host `swift test` runs on (this package's Package.swift
// declares both platforms). No asset catalog is needed in a SwiftPM target.
extension Color {
    init(light: Color, dark: Color) {
        #if canImport(UIKit)
        self.init(UIColor { traits in
            traits.userInterfaceStyle == .dark ? UIColor(dark) : UIColor(light)
        })
        #elseif canImport(AppKit)
        self.init(NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            return NSColor(isDark ? dark : light)
        })
        #else
        self = light
        #endif
    }

    init(light hex: UInt32, dark: UInt32) {
        self.init(light: Color(hex: hex), dark: Color(hex: dark))
    }

    /// 0xRRGGBB (opaque) — matches the hex literals in the prototype CSS.
    init(hex: UInt32) {
        let r = Double((hex >> 16) & 0xFF) / 255
        let g = Double((hex >> 8) & 0xFF) / 255
        let b = Double(hex & 0xFF) / 255
        self.init(.sRGB, red: r, green: g, blue: b, opacity: 1)
    }
}

/// Design tokens extracted from `docs/design/prototype/seecal-prototype.html`
/// (`:root` custom properties, light block + `@media (prefers-color-scheme: dark)` /
/// `:root[data-theme="dark"]` blocks). Per spec §9, the prototype is the binding
/// visual reference — these values are transcribed verbatim, not approximated.
public enum Theme {
    // MARK: Brand

    /// `--basil` — light #1F7A52 / dark #43C98B.
    public static let basil = Color(light: 0x1F7A52, dark: 0x43C98B)
    /// `--basil-soft` — light #E3EFE8 / dark #12281D.
    public static let basilSoft = Color(light: 0xE3EFE8, dark: 0x12281D)

    // MARK: Macros

    /// `--protein` — light #C2495E / dark #E0728A.
    public static let protein = Color(light: 0xC2495E, dark: 0xE0728A)
    /// `--fat` — light #D99A2B / dark #E6B45A.
    public static let fat = Color(light: 0xD99A2B, dark: 0xE6B45A)
    /// `--carbs` — light #4E79C7 / dark #7D9FE0.
    public static let carbs = Color(light: 0x4E79C7, dark: 0x7D9FE0)

    // MARK: Status

    /// `--danger` — light #C0392B / dark #E4685A.
    public static let danger = Color(light: 0xC0392B, dark: 0xE4685A)

    // MARK: Surfaces / ink

    /// `--app-bg` — light #F6F7F5 / dark #000000.
    public static let appBg = Color(light: 0xF6F7F5, dark: 0x000000)
    /// `--app-card` — light #FFFFFF / dark #1C1C1E.
    public static let appCard = Color(light: 0xFFFFFF, dark: 0x1C1C1E)
    /// `--app-ink` — light #141715 / dark #F2F4F1.
    public static let appInk = Color(light: 0x141715, dark: 0xF2F4F1)
    /// `--app-ink-2` — light #6D746F / dark #9BA39C.
    public static let appInk2 = Color(light: 0x6D746F, dark: 0x9BA39C)
    /// `--app-line` — light #E7EAE6 / dark #2A2C29.
    public static let appLine = Color(light: 0xE7EAE6, dark: 0x2A2C29)

    /// Base color for `--app-elev`'s box-shadow (light `rgba(20,23,21,…)` i.e.
    /// #141715 == light `appInk`; dark `rgba(0,0,0,…)` — note dark mode shadows
    /// stay black, they do NOT reuse dark `appInk` (#F2F4F1), which is a light
    /// color meant for foreground text, not a shadow tint).
    public static let shadowColor = Color(light: 0x141715, dark: 0x000000)

    // MARK: Layout constants (prototype CSS)

    /// `.card{border-radius:18px}`.
    public static let cardRadius: CGFloat = 18
    /// `.tabbar{height:88px}`.
    public static let tabBarHeight: CGFloat = 88
    /// `.tab.scan .cam{width:64px;height:64px;margin-top:-36px}` — the FAB overhangs
    /// the bar by 36pt.
    public static let scanFABOverhang: CGFloat = 36
    /// `.tab.scan .cam{width:64px;height:64px}`.
    public static let scanFABDiameter: CGFloat = 64
    /// `.scr{padding:66px 0 136px}` — bottom scroll clearance so content never sits
    /// under the tab bar + FAB overhang.
    public static let screenBottomInset: CGFloat = 136
}

// MARK: - Card elevation

// `--app-elev` is a two-layer CSS box-shadow (a tight contact shadow plus a soft
// ambient one). SwiftUI's `.shadow` only draws one layer per call, so this applies
// both layers to approximate the same look in both themes.
public extension View {
    func cardElevation() -> some View {
        self
            .shadow(color: Theme.shadowColor.opacity(0.06), radius: 1, x: 0, y: 1)
            .shadow(color: Theme.shadowColor.opacity(0.05), radius: 12, x: 0, y: 8)
    }
}
