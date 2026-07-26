import SwiftUI

// MARK: - Card

/// `.card{background:var(--app-card); border-radius:18px; box-shadow:var(--app-elev); padding:16px;}`
/// Outer margins (`margin:12px 18px 0` in the prototype) are the caller's job — screens
/// stack cards in a `VStack(spacing: 12)` padded `.horizontal, 18`, matching the CSS flow
/// layout rather than baking margins into every card instance.
public struct Card<Content: View>: View {
    private let padding: CGFloat
    private let content: Content

    /// - Parameter padding: matches `.card{padding:16px}` by default. Pass `2` for
    ///   the `.mealcard` variant (`.mealcard{padding:2px 2px}`), which wraps a list
    ///   of `.meal` rows that carry their own `12px 14px` padding.
    public init(padding: CGFloat = 16, @ViewBuilder content: () -> Content) {
        self.padding = padding
        self.content = content()
    }

    public var body: some View {
        content
            .padding(padding)
            .background(Theme.appCard)
            .clipShape(RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous))
            .cardElevation()
    }
}

// MARK: - Section label

/// `.sectionlabel{font-size:13px; font-weight:700; color:var(--app-ink-2); text-transform:uppercase; letter-spacing:.06em}`
public struct SectionLabel: View {
    private let text: String

    public init(_ text: String) {
        self.text = text
    }

    public var body: some View {
        Text(text.uppercased())
            .font(.system(size: 13, weight: .bold))
            .tracking(0.78) // .06em of 13px
            .foregroundStyle(Theme.appInk2)
    }
}

// MARK: - Page header

/// `.pagehead .sub` (uppercase, letter-spaced, ink2) + `.pagehead h2` (30px, tight
/// tracking, bold).
public struct PageHeader: View {
    private let subtitle: String
    private let title: String

    public init(subtitle: String, title: String) {
        self.subtitle = subtitle
        self.title = title
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(subtitle.uppercased())
                .font(.system(size: 13, weight: .semibold))
                .tracking(0.78)
                .foregroundStyle(Theme.appInk2)
            Text(title)
                .font(.system(size: 30, weight: .bold))
                .tracking(-0.6)
                .foregroundStyle(Theme.appInk)
        }
    }
}

// MARK: - Chip button + chips row

/// `.chipbtn` — pill button; selected state uses `basil-soft` background / `basil`
/// text (bold, no border); unselected uses `app-bg` background / `app-ink-2` text
/// with an `app-line` border.
public struct ChipButton: View {
    private let title: String
    private let isSelected: Bool
    private let action: () -> Void

    public init(_ title: String, isSelected: Bool, action: @escaping () -> Void) {
        self.title = title
        self.isSelected = isSelected
        self.action = action
    }

    public var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 12.5, weight: isSelected ? .bold : .semibold))
                .foregroundStyle(isSelected ? Theme.basil : Theme.appInk2)
                .padding(.vertical, 6)
                .padding(.horizontal, 12)
                .background(isSelected ? Theme.basilSoft : Theme.appBg)
                .clipShape(Capsule())
                .overlay(
                    Capsule().stroke(isSelected ? Color.clear : Theme.appLine, lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
    }
}

/// `.chips{display:flex; gap:6px; flex-wrap:wrap}` — a wrapping row of `ChipButton`s.
/// Implemented as a `Layout` (iOS 17+, matching this package's deployment target)
/// rather than a fixed `HStack` so a long option list (e.g. onboarding activity
/// chips) wraps instead of overflowing, same as the CSS flex-wrap.
public struct ChipsFlowLayout: Layout {
    public var spacing: CGFloat = 6

    public init(spacing: CGFloat = 6) {
        self.spacing = spacing
    }

    public func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var rowWidth: CGFloat = 0
        var totalWidth: CGFloat = 0
        var totalHeight: CGFloat = 0
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if rowWidth > 0, rowWidth + spacing + size.width > maxWidth {
                totalHeight += rowHeight + spacing
                totalWidth = max(totalWidth, rowWidth)
                rowWidth = 0
                rowHeight = 0
            }
            rowWidth += (rowWidth > 0 ? spacing : 0) + size.width
            rowHeight = max(rowHeight, size.height)
        }
        totalHeight += rowHeight
        totalWidth = max(totalWidth, rowWidth)
        return CGSize(width: maxWidth.isFinite ? maxWidth : totalWidth, height: totalHeight)
    }

    public func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let maxWidth = bounds.width
        var x: CGFloat = bounds.minX
        var y: CGFloat = bounds.minY
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > bounds.minX, x - bounds.minX + size.width > maxWidth {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), anchor: .topLeading, proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}

/// Convenience wrapper pairing `ChipsFlowLayout` with a `ForEach`, e.g.
/// `ChipsRow(ActivityLevel.allCases) { level in ChipButton(title(level), isSelected: ...) { ... } }`.
public struct ChipsRow<Data: RandomAccessCollection, RowContent: View>: View where Data.Element: Hashable {
    private let data: Data
    private let spacing: CGFloat
    private let content: (Data.Element) -> RowContent

    public init(_ data: Data, spacing: CGFloat = 6, @ViewBuilder content: @escaping (Data.Element) -> RowContent) {
        self.data = data
        self.spacing = spacing
        self.content = content
    }

    public var body: some View {
        ChipsFlowLayout(spacing: spacing) {
            ForEach(Array(data), id: \.self) { element in
                content(element)
            }
        }
    }
}

// MARK: - Gram stepper

/// `.stepper` — bordered, rounded, `−` / value / `＋`, tabular numerals. Generic
/// over the formatted value label so it serves height/weight (onboarding, profile)
/// and per-item grams (result/edit sheet) alike.
public struct GramStepper: View {
    private let valueText: String
    private let unitText: String
    private let onDecrement: () -> Void
    private let onIncrement: () -> Void
    private let decrementLabel: String
    private let incrementLabel: String

    public init(
        valueText: String,
        unitText: String,
        decrementLabel: String = "Decrease",
        incrementLabel: String = "Increase",
        onDecrement: @escaping () -> Void,
        onIncrement: @escaping () -> Void
    ) {
        self.valueText = valueText
        self.unitText = unitText
        self.decrementLabel = decrementLabel
        self.incrementLabel = incrementLabel
        self.onDecrement = onDecrement
        self.onIncrement = onIncrement
    }

    public var body: some View {
        HStack(spacing: 0) {
            stepButton("−", action: onDecrement, accessibilityLabel: decrementLabel)
            (
                Text(valueText)
                    .font(.system(size: 13.5, weight: .bold))
                    .foregroundStyle(Theme.appInk)
                + Text(" " + unitText)
                    .font(.system(size: 13.5, weight: .semibold))
                    .foregroundStyle(Theme.appInk2)
            )
            .monospacedDigit()
            .frame(width: 56)
            stepButton("＋", action: onIncrement, accessibilityLabel: incrementLabel)
        }
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(Theme.appLine, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private func stepButton(_ symbol: String, action: @escaping () -> Void, accessibilityLabel: String) -> some View {
        Button(action: action) {
            Text(symbol)
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(Theme.basil)
                .frame(width: 32, height: 32)
                .background(Theme.appBg)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
    }
}

// MARK: - Privacy chip

/// `.privchip{background:var(--basil-soft); color:var(--basil)}` — brand-tinted
/// pill with a lock glyph, e.g. "Analyzed entirely on this iPhone."
public struct PrivacyChip: View {
    private let text: String

    public init(_ text: String) {
        self.text = text
    }

    public var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "lock.fill")
                .font(.system(size: 14, weight: .semibold))
            Text(text)
                .font(.system(size: 12.5, weight: .semibold))
                .fixedSize(horizontal: false, vertical: true)
        }
        .foregroundStyle(Theme.basil)
        .padding(.vertical, 12)
        .padding(.horizontal, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.basilSoft)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}
