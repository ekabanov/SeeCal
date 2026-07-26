import XCTest
@testable import SeeCalApp

/// Pins the History chart's vertical geometry (spec §6): bars are anchored to
/// the BOTTOM of the chart area and the goal line measures from the same
/// bottom edge — the regression this guards against rendered the chart
/// upside-down (bars hanging from the top while the goal line measured from
/// the bottom).
final class HistoryChartLayoutTests: XCTestCase {
    private let height: CGFloat = 150

    func testFullValueBarFillsChartFromBottomToTop() {
        let h = HistoryChartLayout.barHeight(value: 2400, axisMax: 2400, totalHeight: height)
        XCTAssertEqual(h, height)
        XCTAssertEqual(
            HistoryChartLayout.barTopY(value: 2400, axisMax: 2400, totalHeight: height),
            0,
            "A max-value bar's top edge must reach the top of the chart"
        )
    }

    func testHalfValueBarOccupiesLowerHalf() {
        let h = HistoryChartLayout.barHeight(value: 1200, axisMax: 2400, totalHeight: height)
        XCTAssertEqual(h, height / 2)
        XCTAssertEqual(
            HistoryChartLayout.barTopY(value: 1200, axisMax: 2400, totalHeight: height),
            height / 2,
            "Bars grow UP from the bottom: a half-value bar's top edge sits at mid-height"
        )
    }

    func testBarTopAndGoalLineShareTheSameCoordinateConvention() {
        // A bar whose value equals the goal must have its top edge exactly on
        // the goal line — this is what breaks when bars anchor to the top
        // while the goal line measures from the bottom.
        let goal = 2200.0
        let axisMax = goal * 1.08
        let barTop = HistoryChartLayout.barTopY(value: goal, axisMax: axisMax, totalHeight: height)
        let lineY = HistoryChartLayout.goalLineY(goalCalories: goal, axisMax: axisMax, totalHeight: height)
        XCTAssertEqual(barTop, lineY, accuracy: 0.0001)
    }

    func testGoalLineIsHigherOnScreenForBiggerGoals() {
        // y is measured from the top, so a bigger goal means a SMALLER y.
        let low = HistoryChartLayout.goalLineY(goalCalories: 1000, axisMax: 3000, totalHeight: height)
        let high = HistoryChartLayout.goalLineY(goalCalories: 2500, axisMax: 3000, totalHeight: height)
        XCTAssertLessThan(high, low)
        XCTAssertEqual(HistoryChartLayout.goalLineY(goalCalories: 3000, axisMax: 3000, totalHeight: height), 0)
    }

    func testUnloggedAndZeroValuesKeepMinimumSliver() {
        XCTAssertEqual(
            HistoryChartLayout.barHeight(value: 0, axisMax: 2400, totalHeight: height),
            HistoryChartLayout.minBarHeight
        )
        // Sliver still bottom-anchored.
        XCTAssertEqual(
            HistoryChartLayout.barTopY(value: 0, axisMax: 2400, totalHeight: height),
            height - HistoryChartLayout.minBarHeight
        )
        // Tiny non-zero values get the 4pt minimum.
        XCTAssertEqual(
            HistoryChartLayout.barHeight(value: 1, axisMax: 2400, totalHeight: height),
            HistoryChartLayout.minValueBarHeight
        )
        // Degenerate axis never divides by zero.
        XCTAssertEqual(
            HistoryChartLayout.barHeight(value: 500, axisMax: 0, totalHeight: height),
            HistoryChartLayout.minBarHeight
        )
    }
}
