import Foundation
import SeeCalDomain
import SeeCalPersistence

public enum BarcodeSymbology: String, Codable, Sendable {
    case ean8
    case ean13
    case upca
    case upce
    case gtin14
    case code128
    case unknown
}

public struct DetectedBarcode: Equatable, Sendable {
    public var value: String
    public var symbology: BarcodeSymbology

    public init(value: String, symbology: BarcodeSymbology = .unknown) {
        self.value = value
        self.symbology = symbology
    }
}

public enum GTIN {
    /// Removes presentation separators, expands UPC-E when its symbology is
    /// known, and validates the GS1 check digit.
    public static func normalized(
        _ rawValue: String,
        symbology: BarcodeSymbology = .unknown
    ) -> String? {
        let digits = rawValue.filter(\.isNumber)
        guard digits.count == rawValue.filter({ !$0.isWhitespace && $0 != "-" }).count else {
            return nil
        }

        let candidate: String
        if symbology == .upce {
            guard let expanded = expandUPCE(digits) else { return nil }
            candidate = expanded
        } else {
            candidate = digits
        }

        guard [8, 12, 13, 14].contains(candidate.count),
              candidate.allSatisfy(\.isNumber),
              hasValidCheckDigit(candidate)
        else { return nil }
        return candidate
    }

    public static func hasValidCheckDigit(_ digits: String) -> Bool {
        guard digits.count >= 2,
              let check = digits.last?.wholeNumberValue
        else { return false }

        let body = digits.dropLast().reversed()
        let sum = body.enumerated().reduce(0) { partial, pair in
            guard let digit = pair.element.wholeNumberValue else { return Int.max }
            return partial + digit * (pair.offset.isMultiple(of: 2) ? 3 : 1)
        }
        return (10 - (sum % 10)) % 10 == check
    }

    /// UPC-E number-system 0/1 expansion into the UPC-A representation used by
    /// product databases. Input includes the number-system and check digits.
    private static func expandUPCE(_ digits: String) -> String? {
        guard digits.count == 8 else { return nil }
        let values = digits.compactMap(\.wholeNumberValue)
        guard values.count == 8, values[0] == 0 || values[0] == 1 else { return nil }

        let ns = values[0]
        let d1 = values[1], d2 = values[2], d3 = values[3]
        let d4 = values[4], d5 = values[5], d6 = values[6]
        let check = values[7]
        let body: [Int]
        switch d6 {
        case 0...2:
            body = [ns, d1, d2, d6, 0, 0, 0, 0, d3, d4, d5]
        case 3:
            body = [ns, d1, d2, d3, 0, 0, 0, 0, 0, d4, d5]
        case 4:
            body = [ns, d1, d2, d3, d4, 0, 0, 0, 0, 0, d5]
        default:
            body = [ns, d1, d2, d3, d4, d5, 0, 0, 0, 0, d6]
        }
        return (body + [check]).map(String.init).joined()
    }
}

public struct BarcodeNutrition: Codable, Equatable, Sendable {
    public var kcal: Double
    public var protein: Double
    public var fat: Double
    public var carbs: Double

    public init(kcal: Double, protein: Double, fat: Double, carbs: Double) {
        self.kcal = kcal
        self.protein = protein
        self.fat = fat
        self.carbs = carbs
    }

    public func scaled(to amount: Double) -> BarcodeNutrition {
        let factor = amount / 100
        return BarcodeNutrition(
            kcal: kcal * factor,
            protein: protein * factor,
            fat: fat * factor,
            carbs: carbs * factor
        )
    }
}

public struct BarcodeProduct: Codable, Equatable, Sendable {
    public var barcode: String
    public var name: String
    public var ingredientsText: String?
    public var servingDescription: String?
    public var defaultAmount: Double?
    public var amountUnit: MealAmountUnit
    public var kcalPer100: Double?
    public var proteinPer100: Double?
    public var fatPer100: Double?
    public var carbsPer100: Double?
    public var provider: String
    public var lookedUpAt: Date

    public init(
        barcode: String,
        name: String,
        ingredientsText: String? = nil,
        servingDescription: String? = nil,
        defaultAmount: Double? = nil,
        amountUnit: MealAmountUnit = .grams,
        kcalPer100: Double?,
        proteinPer100: Double?,
        fatPer100: Double?,
        carbsPer100: Double?,
        provider: String = "Open Food Facts",
        lookedUpAt: Date = Date()
    ) {
        self.barcode = barcode
        self.name = name
        self.ingredientsText = ingredientsText
        self.servingDescription = servingDescription
        self.defaultAmount = defaultAmount
        self.amountUnit = amountUnit
        self.kcalPer100 = kcalPer100
        self.proteinPer100 = proteinPer100
        self.fatPer100 = fatPer100
        self.carbsPer100 = carbsPer100
        self.provider = provider
        self.lookedUpAt = lookedUpAt
    }

    public var nutritionPer100: BarcodeNutrition? {
        guard let kcalPer100, let proteinPer100, let fatPer100, let carbsPer100 else {
            return nil
        }
        return BarcodeNutrition(
            kcal: kcalPer100,
            protein: proteinPer100,
            fat: fatPer100,
            carbs: carbsPer100
        )
    }

    public func mealDraft(amount: Double, mealType: MealType) -> MealEditDraft? {
        guard amount > 0, let nutritionPer100 else { return nil }
        let scaled = nutritionPer100.scaled(to: amount)
        let base = MealItemBase(
            grams: amount,
            kcal: scaled.kcal,
            protein: scaled.protein,
            fat: scaled.fat,
            carbs: scaled.carbs
        )
        let item = MealItem(
            name: name,
            grams: amount,
            amountUnit: amountUnit,
            base: base,
            sourceReference: MealItemReference(name: name, base: base, source: .barcode)
        )
        let source = BarcodeSourceMetadata(
            barcode: barcode,
            provider: provider,
            lookedUpAt: lookedUpAt,
            productName: name,
            ingredientsText: ingredientsText,
            servingDescription: servingDescription
        )
        return MealEditDraft(
            barcodeItem: item,
            name: name,
            mealType: mealType,
            source: source
        )
    }
}

public enum BarcodeLookupError: Error, Equatable, LocalizedError, Sendable {
    case invalidBarcode
    case notFound
    case invalidResponse
    case requestFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidBarcode:
            return "That barcode is not valid."
        case .notFound:
            return "Product not found in Open Food Facts."
        case .invalidResponse:
            return "Open Food Facts returned an unreadable product record."
        case let .requestFailed(message):
            return "Barcode lookup failed: \(message)"
        }
    }
}

public protocol BarcodeProductLookup: Sendable {
    func lookup(barcode: DetectedBarcode) async throws -> BarcodeProduct
}

public protocol BarcodeProductCaching: Sendable {
    func product(for barcode: String) async -> BarcodeProduct?
    func store(_ product: BarcodeProduct) async throws
}

public actor InMemoryBarcodeProductCache: BarcodeProductCaching {
    private var products: [String: BarcodeProduct]

    public init(products: [String: BarcodeProduct] = [:]) {
        self.products = products
    }

    public func product(for barcode: String) -> BarcodeProduct? {
        products[barcode]
    }

    public func store(_ product: BarcodeProduct) {
        products[product.barcode] = product
    }
}

public actor FileBackedBarcodeProductCache: BarcodeProductCaching {
    private let fileURL: URL
    private var products: [String: BarcodeProduct]

    public init(fileURL: URL? = nil) {
        self.fileURL = fileURL ?? FileBackedStoreLocations.applicationSupportDirectory
            .appendingPathComponent("barcode_products.json")
        if let data = try? Data(contentsOf: self.fileURL),
           let decoded = try? JSONDecoder.iso8601.decode([String: BarcodeProduct].self, from: data) {
            self.products = decoded
        } else {
            self.products = [:]
        }
    }

    public func product(for barcode: String) -> BarcodeProduct? {
        products[barcode]
    }

    public func store(_ product: BarcodeProduct) throws {
        products[product.barcode] = product
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = try JSONEncoder.iso8601.encode(products)
        try data.write(to: fileURL, options: .atomic)
    }
}

public actor CachedBarcodeProductLookup: BarcodeProductLookup {
    private let remote: BarcodeProductLookup
    private let cache: BarcodeProductCaching

    public init(
        remote: BarcodeProductLookup = RateLimitedBarcodeProductLookup(
            remote: OpenFoodFactsProductLookup()
        ),
        cache: BarcodeProductCaching = FileBackedBarcodeProductCache()
    ) {
        self.remote = remote
        self.cache = cache
    }

    public func lookup(barcode: DetectedBarcode) async throws -> BarcodeProduct {
        guard let normalized = GTIN.normalized(barcode.value, symbology: barcode.symbology) else {
            throw BarcodeLookupError.invalidBarcode
        }
        if let cached = await cache.product(for: normalized) {
            return cached
        }
        let product = try await remote.lookup(
            barcode: DetectedBarcode(value: normalized, symbology: barcode.symbology)
        )
        try await cache.store(product)
        return product
    }
}

/// Serializes remote reads to the provider's documented product-read budget.
/// The cache wraps this type, so a local hit never waits.
public actor RateLimitedBarcodeProductLookup: BarcodeProductLookup {
    public typealias Sleeper = @Sendable (TimeInterval) async throws -> Void

    private let remote: BarcodeProductLookup
    private let minimumIntervalSeconds: TimeInterval
    private let now: @Sendable () -> Date
    private let sleeper: Sleeper
    private var nextAllowedAt: Date?

    public init(
        remote: BarcodeProductLookup,
        minimumIntervalSeconds: TimeInterval = 4,
        now: @escaping @Sendable () -> Date = Date.init,
        sleeper: @escaping Sleeper = { seconds in
            try await Task<Never, Never>.sleep(
                nanoseconds: UInt64(max(0, seconds) * 1_000_000_000)
            )
        }
    ) {
        self.remote = remote
        self.minimumIntervalSeconds = max(0, minimumIntervalSeconds)
        self.now = now
        self.sleeper = sleeper
    }

    public func lookup(barcode: DetectedBarcode) async throws -> BarcodeProduct {
        let current = now()
        let scheduled = max(current, nextAllowedAt ?? current)
        // Reserve before suspending so a re-entrant caller receives the next
        // slot instead of sharing this one.
        nextAllowedAt = scheduled.addingTimeInterval(minimumIntervalSeconds)

        let delay = scheduled.timeIntervalSince(current)
        if delay > 0 {
            try await sleeper(delay)
        }
        try Task.checkCancellation()
        return try await remote.lookup(barcode: barcode)
    }
}

public protocol BarcodeHTTPDataLoading: Sendable {
    func data(for request: URLRequest) async throws -> (Data, URLResponse)
}

extension URLSession: BarcodeHTTPDataLoading {}

public struct OpenFoodFactsProductLookup: BarcodeProductLookup {
    private let session: BarcodeHTTPDataLoading
    private let baseURL: URL
    private let now: @Sendable () -> Date

    public init(
        session: BarcodeHTTPDataLoading = URLSession.shared,
        baseURL: URL = URL(string: "https://world.openfoodfacts.org")!,
        now: @escaping @Sendable () -> Date = Date.init
    ) {
        self.session = session
        self.baseURL = baseURL
        self.now = now
    }

    public func lookup(barcode: DetectedBarcode) async throws -> BarcodeProduct {
        guard let normalized = GTIN.normalized(barcode.value, symbology: barcode.symbology) else {
            throw BarcodeLookupError.invalidBarcode
        }

        var components = URLComponents(
            url: baseURL.appending(path: "api/v3/product/\(normalized)"),
            resolvingAgainstBaseURL: false
        )
        components?.queryItems = [
            URLQueryItem(
                name: "fields",
                value: [
                    "code", "product_name", "ingredients_text", "serving_size",
                    "serving_quantity", "serving_quantity_unit",
                    "product_quantity_unit", "nutriments"
                ].joined(separator: ",")
            )
        ]
        guard let url = components?.url else { throw BarcodeLookupError.invalidResponse }

        var request = URLRequest(url: url)
        request.setValue(
            "SeeCal/0.1 (https://github.com/ekabanov/SeeCal)",
            forHTTPHeaderField: "User-Agent"
        )
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw BarcodeLookupError.invalidResponse
            }
            if http.statusCode == 404 { throw BarcodeLookupError.notFound }
            guard (200..<300).contains(http.statusCode) else {
                throw BarcodeLookupError.requestFailed("HTTP \(http.statusCode)")
            }
            let payload = try JSONDecoder().decode(OpenFoodFactsResponse.self, from: data)
            guard let product = payload.product else { throw BarcodeLookupError.notFound }
            return product.normalized(barcode: normalized, lookedUpAt: now())
        } catch let error as BarcodeLookupError {
            throw error
        } catch {
            throw BarcodeLookupError.requestFailed(error.localizedDescription)
        }
    }
}

private struct OpenFoodFactsResponse: Decodable {
    var product: OpenFoodFactsProduct?
}

private struct OpenFoodFactsProduct: Decodable {
    var productName: String?
    var ingredientsText: String?
    var servingSize: String?
    var servingQuantity: FlexibleDouble?
    var servingQuantityUnit: String?
    var productQuantityUnit: String?
    var nutriments: OpenFoodFactsNutriments?

    enum CodingKeys: String, CodingKey {
        case productName = "product_name"
        case ingredientsText = "ingredients_text"
        case servingSize = "serving_size"
        case servingQuantity = "serving_quantity"
        case servingQuantityUnit = "serving_quantity_unit"
        case productQuantityUnit = "product_quantity_unit"
        case nutriments
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        // Open Food Facts is community-authored. Treat malformed optional fields
        // as absent so one bad serving value cannot hide an otherwise usable
        // product record.
        productName = try? container.decode(String.self, forKey: .productName)
        ingredientsText = try? container.decode(String.self, forKey: .ingredientsText)
        servingSize = try? container.decode(String.self, forKey: .servingSize)
        servingQuantity = try? container.decode(FlexibleDouble.self, forKey: .servingQuantity)
        servingQuantityUnit = try? container.decode(String.self, forKey: .servingQuantityUnit)
        productQuantityUnit = try? container.decode(String.self, forKey: .productQuantityUnit)
        nutriments = try? container.decode(OpenFoodFactsNutriments.self, forKey: .nutriments)
    }

    func normalized(barcode: String, lookedUpAt: Date) -> BarcodeProduct {
        let quantity = Self.normalizedQuantity(
            value: servingQuantity?.value,
            unit: servingQuantityUnit ?? productQuantityUnit
        )
        let unit = quantity?.unit ?? Self.amountUnit(for: productQuantityUnit ?? servingQuantityUnit)
        return BarcodeProduct(
            barcode: barcode,
            name: productName?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
                ?? "Packaged food",
            ingredientsText: ingredientsText?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty,
            servingDescription: servingSize?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty,
            defaultAmount: quantity?.amount,
            amountUnit: unit,
            kcalPer100: nutriments?.kcalPer100?.value,
            proteinPer100: nutriments?.proteinPer100?.value,
            fatPer100: nutriments?.fatPer100?.value,
            carbsPer100: nutriments?.carbsPer100?.value,
            lookedUpAt: lookedUpAt
        )
    }

    private static func amountUnit(for rawUnit: String?) -> MealAmountUnit {
        switch rawUnit?.lowercased() {
        case "ml", "cl", "dl", "l":
            return .milliliters
        default:
            return .grams
        }
    }

    private static func normalizedQuantity(
        value: Double?,
        unit rawUnit: String?
    ) -> (amount: Double, unit: MealAmountUnit)? {
        guard let value, value > 0, let rawUnit = rawUnit?.lowercased() else { return nil }
        switch rawUnit {
        case "g": return (value, .grams)
        case "kg": return (value * 1_000, .grams)
        case "ml": return (value, .milliliters)
        case "cl": return (value * 10, .milliliters)
        case "dl": return (value * 100, .milliliters)
        case "l": return (value * 1_000, .milliliters)
        default: return nil
        }
    }
}

/// Decode only the nutrition fields SeeCal consumes. A real `nutriments`
/// object also contains strings such as `sodium_modifier: "~"` and many other
/// heterogeneous values; decoding the whole object as numeric values rejects
/// valid production records.
private struct OpenFoodFactsNutriments: Decodable {
    var kcalPer100: FlexibleDouble?
    var proteinPer100: FlexibleDouble?
    var fatPer100: FlexibleDouble?
    var carbsPer100: FlexibleDouble?

    enum CodingKeys: String, CodingKey {
        case kcalPer100 = "energy-kcal_100g"
        case proteinPer100 = "proteins_100g"
        case fatPer100 = "fat_100g"
        case carbsPer100 = "carbohydrates_100g"
    }
}

private struct FlexibleDouble: Decodable {
    let value: Double

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let double = try? container.decode(Double.self) {
            value = double
        } else if let string = try? container.decode(String.self), let double = Double(string) {
            value = double
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Expected a number or numeric string"
            )
        }
    }
}

private extension JSONEncoder {
    static var iso8601: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}

private extension JSONDecoder {
    static var iso8601: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}

private extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
