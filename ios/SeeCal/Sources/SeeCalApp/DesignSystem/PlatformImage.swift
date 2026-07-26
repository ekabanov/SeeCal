import SwiftUI

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

/// Loads a meal photo from disk into a SwiftUI `Image`, on whichever platform
/// this package is built for (UIKit on iOS, AppKit on the macOS test host).
/// Shared by the Today meal rows, the Analyzing screen's shot card, and the
/// result sheet's header thumbnail.
enum PlatformImageLoader {
    static func image(atPath path: String) -> Image? {
        guard !path.isEmpty, FileManager.default.fileExists(atPath: path) else { return nil }
        #if canImport(UIKit)
        guard let uiImage = UIImage(contentsOfFile: path) else { return nil }
        return Image(uiImage: uiImage)
        #elseif canImport(AppKit)
        guard let nsImage = NSImage(contentsOfFile: path) else { return nil }
        return Image(nsImage: nsImage)
        #else
        return nil
        #endif
    }
}
