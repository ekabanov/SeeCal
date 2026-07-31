import Foundation
import SQLite3
import XCTest
@testable import SeeCalDomain
@testable import SeeCalInference

private struct StubNutritionResolver: NutritionResolving {
    let resolutions: [String: NutritionResolution]

    func resolve(name: String) async throws -> NutritionResolution {
        resolutions[name] ?? NutritionResolution(
            query: name,
            rung: .unresolved,
            score: 0,
            profile: nil,
            estimated: true
        )
    }
}

private struct StubFoodIdentifier: FoodIdentifying {
    let identification: FoodIdentification

    func identify(request: FoodScanRequest) async throws -> FoodIdentification {
        identification
    }
}

private struct StubScalePredictor: ScalePredicting {
    let prediction: FactoredScalePrediction

    func predict(imagePath: String) async throws -> FactoredScalePrediction {
        prediction
    }
}

final class FactoredNutritionPipelineTests: XCTestCase {
    func testPromptMatchesFrozenPythonContract() {
        XCTAssertEqual(
            FactoredIdentificationPrompt.text,
            "Identify only visually distinguishable food components without estimating " +
                "grams, calories, nutrients, seasonings, or hidden recipe ingredients. " +
                "Return exactly one JSON object with keys not_food, container, and items. " +
                "container must be plate, bowl, cup, tray, packaging, or other. " +
                "items must contain objects with name and portion_units only. " +
                "portion_units must be an integer from 1 to 20 expressing relative visible " +
                "portion size; the values do not need to sum to any particular number. " +
                "For a non-food image return not_food true, container other, and an empty items list."
        )
    }

    func testStrictParserAcceptsContractAndRejectsExtraNutrition() throws {
        let parsed = try IdentificationJSONParser.parseStrict(
            """
            {"not_food":false,"container":"bowl","items":[
              {"name":"rice","portion_units":3},
              {"name":"beans","portion_units":1}
            ]}
            """
        )
        XCTAssertEqual(parsed.container, .bowl)
        XCTAssertEqual(parsed.items.map(\.sharePercent), [75, 25])

        XCTAssertThrowsError(
            try IdentificationJSONParser.parseStrict(
                """
                {"not_food":false,"container":"bowl","items":[
                  {"name":"rice","portion_units":20,"calories":130}
                ]}
                """
            )
        )
    }

    func testParserMergesDuplicateNamesAndNormalizesArbitraryUnits() throws {
        let parsed = try IdentificationJSONParser.parseStrict(
            """
            {"not_food":false,"container":"plate","items":[
              {"name":"rice","portion_units":2},
              {"name":"beans","portion_units":4},
              {"name":"Rice","portion_units":3}
            ]}
            """
        )
        XCTAssertEqual(parsed.items.map(\.name), ["rice", "beans"])
        XCTAssertEqual(parsed.items.map(\.sharePercent), [55, 45])
        XCTAssertEqual(parsed.items.map(\.sharePercent).reduce(0, +), 100)
    }

    func testLegacyPercentageParserNormalizesInsteadOfRequiringModelArithmetic() throws {
        let parsed = try IdentificationJSONParser.parseLegacySharesNormalized(
            """
            {"not_food":false,"container":"plate","items":[
              {"name":"rice","share_pct":12},
              {"name":"chicken","share_pct":7.5},
              {"name":"Rice","share_pct":3}
            ]}
            """
        )
        XCTAssertEqual(parsed.items.map(\.name), ["rice", "chicken"])
        XCTAssertEqual(parsed.items.map(\.sharePercent).reduce(0, +), 100)
    }

    func testParsersRejectNotFoodWithItems() {
        XCTAssertThrowsError(
            try IdentificationJSONParser.parseStrict(
                """
                {"not_food":true,"container":"other","items":[
                  {"name":"rice","portion_units":1}
                ]}
                """
            )
        )
        XCTAssertThrowsError(
            try IdentificationJSONParser.parseLegacySharesNormalized(
                """
                {"not_food":true,"container":"other","items":[
                  {"name":"rice","share_pct":100}
                ]}
                """
            )
        )
    }

    func testAssemblerUsesLookupAndScaleArithmetic() throws {
        let profile = try ResolvedNutritionProfile(
            fdcID: 1,
            name: "Cucumber, raw",
            category: "Vegetables",
            kcalPer100g: 18.1,
            proteinPer100g: 0.7,
            fatPer100g: 0.1,
            carbsPer100g: 3.6,
            sourceKcalPer100g: 15,
            dataType: "Foundation",
            typicalPortionGrams: 100
        )
        let identification = try FoodIdentification(
            notFood: false,
            container: .plate,
            items: [IdentifiedFoodItem(name: "cucumber", sharePercent: 100)]
        )
        let scale = try FactoredScalePrediction(
            massGrams: .init(estimate: 150, low: 100, high: 200)
        )
        let meal = try FactoredAssembler.assemble(
            identification: identification,
            scale: scale,
            resolutions: [
                .init(
                    query: "cucumber",
                    rung: .exactAlias,
                    score: 1,
                    profile: profile,
                    estimated: false
                )
            ]
        )
        let item = try XCTUnwrap(meal.items.first)
        let nutrition = try XCTUnwrap(item.nutrition)
        XCTAssertEqual(item.grams, .init(estimate: 150, low: 100, high: 200))
        XCTAssertEqual(nutrition.calories.estimate, 27.15, accuracy: 0.0001)
        XCTAssertEqual(nutrition.calories.low, 18.1, accuracy: 0.0001)
        let scan = try XCTUnwrap(meal.foodScanResult)
        XCTAssertEqual(scan.totalCalories, 27.15, accuracy: 0.0001)
        XCTAssertEqual(
            4 * scan.proteinGrams + 9 * scan.fatGrams + 4 * scan.carbsGrams,
            scan.totalCalories,
            accuracy: 0.0001
        )
        XCTAssertTrue(meal.confirmationReasons.isEmpty)
    }

    func testFactoredRuntimeBridgesResolvedMealToScanResult() async throws {
        let identification = try FoodIdentification(
            notFood: false,
            container: .bowl,
            items: [IdentifiedFoodItem(name: "rice", sharePercent: 100)]
        )
        let scale = try FactoredScalePrediction(
            massGrams: .init(estimate: 200, low: 150, high: 250)
        )
        let profile = try ResolvedNutritionProfile(
            fdcID: 1,
            name: "Rice, cooked",
            category: "Grains",
            kcalPer100g: 130,
            proteinPer100g: 2.7,
            fatPer100g: 0.3,
            carbsPer100g: 28,
            dataType: "fixture"
        )
        let runtime = FactoredNutritionRuntime(
            pipeline: FactoredNutritionInferencePipeline(
                identifier: StubFoodIdentifier(identification: identification),
                scalePredictor: StubScalePredictor(prediction: scale),
                resolver: StubNutritionResolver(
                    resolutions: [
                        "rice": NutritionResolution(
                            query: "rice",
                            rung: .exactAlias,
                            score: 1,
                            profile: profile,
                            estimated: false
                        )
                    ]
                )
            )
        )
        let result = try await runtime.infer(
            request: FoodScanRequest(imagePath: "/tmp/unused.jpg", mealType: .lunch)
        )
        XCTAssertEqual(result.items.map(\.name), ["rice"])
        XCTAssertEqual(result.items[0].estimatedGrams, 200)
        XCTAssertEqual(result.totalCalories, 260)
    }

    func testUnresolvedItemIsVisibleAndNeverBecomesZeros() throws {
        let identification = try FoodIdentification(
            notFood: false,
            container: .tray,
            items: [IdentifiedFoodItem(name: "mystery", sharePercent: 100)]
        )
        let scale = try FactoredScalePrediction(
            massGrams: .init(estimate: 100, low: 50, high: 150)
        )
        let meal = try FactoredAssembler.assemble(
            identification: identification,
            scale: scale,
            resolutions: [
                .init(
                    query: "mystery",
                    rung: .unresolved,
                    score: 0,
                    profile: nil,
                    estimated: true
                )
            ]
        )
        XCTAssertNil(meal.items[0].nutrition)
        XCTAssertNil(meal.foodScanResult)
        XCTAssertTrue(meal.confirmationReasons.contains(.unresolvedItem))
    }

    func testScaleDisagreementSumsSelectedPortionsWithoutShareDivision() throws {
        func profile(id: Int64, name: String, portion: Double) throws
            -> ResolvedNutritionProfile
        {
            try ResolvedNutritionProfile(
                fdcID: id,
                name: name,
                category: "Test",
                kcalPer100g: 100,
                proteinPer100g: 10,
                fatPer100g: 0,
                carbsPer100g: 15,
                dataType: "fixture",
                typicalPortionGrams: portion
            )
        }
        let identification = try FoodIdentification(
            notFood: false,
            container: .plate,
            items: [
                IdentifiedFoodItem(name: "pizza slice", sharePercent: 90),
                IdentifiedFoodItem(name: "whole pizza", sharePercent: 10),
            ]
        )
        let resolutions = [
            NutritionResolution(
                query: "pizza slice",
                rung: .exactAlias,
                score: 1,
                profile: try profile(id: 1, name: "Pizza slice", portion: 112),
                estimated: false
            ),
            NutritionResolution(
                query: "whole pizza",
                rung: .exactAlias,
                score: 1,
                profile: try profile(id: 2, name: "Whole pizza", portion: 897),
                estimated: false
            ),
        ]
        let compatibleScale = try FactoredScalePrediction(
            massGrams: .init(estimate: 550, low: 450, high: 600)
        )
        let compatible = try FactoredAssembler.assemble(
            identification: identification,
            scale: compatibleScale,
            resolutions: resolutions
        )
        XCTAssertFalse(compatible.confirmationReasons.contains(.scaleDisagreement))

        let incompatibleScale = try FactoredScalePrediction(
            massGrams: .init(estimate: 225, low: 150, high: 300)
        )
        let incompatible = try FactoredAssembler.assemble(
            identification: identification,
            scale: incompatibleScale,
            resolutions: resolutions
        )
        XCTAssertTrue(incompatible.confirmationReasons.contains(.scaleDisagreement))
    }

    func testSQLiteResolverWalksExactAndFuzzyRungs() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("sqlite")
        defer { try? FileManager.default.removeItem(at: url) }
        var database: OpaquePointer?
        XCTAssertEqual(sqlite3_open(url.path, &database), SQLITE_OK)
        let sql = """
        CREATE TABLE foods(
          fdc_id INTEGER PRIMARY KEY,name TEXT,normalized_name TEXT,category TEXT,
          data_type TEXT,kcal_per_100g REAL,source_kcal_per_100g REAL,
          protein_per_100g REAL,fat_per_100g REAL,carbs_per_100g REAL,
          typical_portion_g REAL);
        CREATE TABLE aliases(
          normalized_alias TEXT,fdc_id INTEGER,priority INTEGER,source TEXT);
        CREATE TABLE category_defaults(
          category TEXT PRIMARY KEY,name TEXT,normalized_name TEXT,
          kcal_per_100g REAL,source_kcal_per_100g REAL,protein_per_100g REAL,
          fat_per_100g REAL,carbs_per_100g REAL,typical_portion_g REAL,
          data_type TEXT);
        INSERT INTO foods VALUES
          (1,'Cucumber, raw','cucumber raw','Vegetables','Foundation',
           18.1,15,0.7,0.1,3.6,100);
        INSERT INTO aliases VALUES('cucumber',1,10,'fdc_description');
        INSERT INTO category_defaults VALUES
          ('Vegetables','Typical Vegetables','typical vegetables',
           50,NULL,2,1,8,100,'category_median');
        """
        XCTAssertEqual(sqlite3_exec(database, sql, nil, nil, nil), SQLITE_OK)
        sqlite3_close(database)

        let resolver = try SQLiteNutritionResolver(databaseURL: url)
        let exact = try await resolver.resolve(name: "Cucumber")
        let fuzzy = try await resolver.resolve(name: "cucumber raw")
        XCTAssertEqual(exact.rung, .exactAlias)
        XCTAssertEqual(fuzzy.rung, .fuzzy)
        XCTAssertEqual(exact.profile?.fdcID, 1)
    }

    func testReplacementCandidatesUseRealProfilesAndExcludeCurrentFood() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("sqlite")
        defer { try? FileManager.default.removeItem(at: url) }
        var database: OpaquePointer?
        XCTAssertEqual(sqlite3_open(url.path, &database), SQLITE_OK)
        let sql = """
        CREATE TABLE foods(
          fdc_id INTEGER PRIMARY KEY,name TEXT,normalized_name TEXT,category TEXT,
          data_type TEXT,kcal_per_100g REAL,source_kcal_per_100g REAL,
          protein_per_100g REAL,fat_per_100g REAL,carbs_per_100g REAL,
          typical_portion_g REAL);
        CREATE TABLE aliases(
          normalized_alias TEXT,fdc_id INTEGER,priority INTEGER,source TEXT);
        CREATE TABLE category_defaults(
          category TEXT PRIMARY KEY,name TEXT,normalized_name TEXT,
          kcal_per_100g REAL,source_kcal_per_100g REAL,protein_per_100g REAL,
          fat_per_100g REAL,carbs_per_100g REAL,typical_portion_g REAL,
          data_type TEXT);
        INSERT INTO foods VALUES
          (1,'Chicken breast','chicken breast','Poultry','Foundation',165,NULL,31,3.6,0,100),
          (2,'Turkey breast','turkey breast','Poultry','Foundation',135,NULL,30,1,0,100),
          (3,'Tofu, firm','tofu firm','Legumes','Foundation',120,NULL,13,7,2,100),
          (4,'Salmon, cooked','salmon cooked','Fish','Foundation',206,NULL,22,12,0,100),
          (5,'Egg, whole','egg whole','Eggs','Foundation',143,NULL,13,10,1,50),
          (6,'Paneer','paneer','Dairy','Foundation',265,NULL,18,21,2,100);
        INSERT INTO aliases VALUES
          ('chicken',1,10,'test'),
          ('turkey',2,10,'test'),
          ('tofu',3,10,'test'),
          ('salmon',4,10,'test'),
          ('egg',5,10,'test'),
          ('paneer',6,10,'test');
        """
        XCTAssertEqual(sqlite3_exec(database, sql, nil, nil, nil), SQLITE_OK)
        sqlite3_close(database)

        let resolver = try SQLiteNutritionResolver(databaseURL: url)
        let candidates = await resolver.candidates(for: "chicken breast", limit: 5)

        XCTAssertEqual(candidates.map(\.displayName), [
            "Turkey", "Tofu", "Salmon", "Egg", "Paneer",
        ])
        XCTAssertEqual(candidates.map(\.profile.fdcID), [2, 3, 4, 5, 6])
        XCTAssertFalse(candidates.contains { $0.profile.fdcID == 1 })
        XCTAssertTrue(candidates.allSatisfy { $0.profile.kcalPer100g > 0 })
    }

    func testShadowPipelineCanExerciseCurrentV8Names() async throws {
        let profile = try ResolvedNutritionProfile(
            fdcID: 1,
            name: "Rice",
            category: "Grains",
            kcalPer100g: 130,
            proteinPer100g: 2.5,
            fatPer100g: 0.3,
            carbsPer100g: 28,
            dataType: "FNDDS"
        )
        let resolution = NutritionResolution(
            query: "rice",
            rung: .exactAlias,
            score: 1,
            profile: profile,
            estimated: false
        )
        let resolver = StubNutritionResolver(resolutions: ["rice": resolution])
        let monolith = FoodScanResult(
            totalCalories: 260,
            proteinGrams: 5,
            fatGrams: 0.6,
            carbsGrams: 56,
            items: [
                ScanItem(
                    name: "rice",
                    estimatedGrams: 200,
                    calories: 260,
                    proteinGrams: 5,
                    fatGrams: 0.6,
                    carbsGrams: 56
                )
            ]
        )
        let interval = VisualSpecialistInterval(estimate: 180, low: 140, high: 220)
        let specialist = VisualSpecialistPrediction(
            massG: interval,
            calories: interval,
            proteinG: interval,
            fatG: interval,
            carbsG: interval
        )
        let meal = try await FactoredShadowPipeline.run(
            monolithResult: monolith,
            specialistResult: specialist,
            resolver: resolver
        )
        XCTAssertEqual(meal.items[0].grams.estimate, 180)
        let scan = try XCTUnwrap(meal.foodScanResult)
        XCTAssertEqual(scan.totalCalories, 234, accuracy: 0.0001)
    }

    func testRealScaleCoreMLArtifactMatchesPythonPredictionWhenPresent() async throws {
        let repository = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // SeeCalInferenceTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // SeeCal
            .deletingLastPathComponent() // ios
            .deletingLastPathComponent() // repository
        let model = repository.appendingPathComponent(
            "ml/runs/factored/scale-v1/deployment/SeeCalScale.mlmodelc",
            isDirectory: true
        )
        let evaluation = repository.appendingPathComponent(
            "ml/runs/factored/scale-v1/test.json"
        )
        guard FileManager.default.fileExists(atPath: model.path),
              FileManager.default.fileExists(atPath: evaluation.path)
        else {
            throw XCTSkip("Local SCALE-v1 artifacts are gitignored")
        }
        let data = try Data(contentsOf: evaluation)
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let paired = try XCTUnwrap(json["paired"] as? [[String: Any]])
        let predictor = try CoreMLScalePredictor(
            modelPath: model.path,
            calibrationMarginGrams: 0.07638931274414062
        )
        var relativeErrors = [Double]()
        var covered = 0
        for row in paired {
            let id = try XCTUnwrap(row["id"] as? String)
            let dishID = id.split(separator: ":")[1]
            let image = repository.appendingPathComponent(
                "ml/dataset_clean/\(dishID)/overhead.jpg"
            )
            let actual = try await predictor.predict(imagePath: image.path).massGrams
            let expected = [
                try XCTUnwrap(row["p10_g"] as? Double),
                try XCTUnwrap(row["p50_g"] as? Double),
                try XCTUnwrap(row["p90_g"] as? Double),
            ]
            for (actualValue, expectedValue) in zip(
                [actual.low, actual.estimate, actual.high],
                expected
            ) {
                relativeErrors.append(
                    abs(actualValue - expectedValue) / max(expectedValue, 1)
                )
            }
            let target = try XCTUnwrap(row["target_mass_g"] as? Double)
            if actual.low <= target, target <= actual.high {
                covered += 1
            }
        }
        let meanRelativeDrift =
            relativeErrors.reduce(0, +) / Double(relativeErrors.count)
        let coverage = Double(covered) / Double(paired.count)
        print(
            "[FactoredNutritionPipelineTests] SCALE Core ML: " +
                "mean relative drift=\(meanRelativeDrift), coverage=\(coverage)"
        )
        XCTAssertLessThan(meanRelativeDrift, 0.10)
        XCTAssertGreaterThanOrEqual(coverage, 0.75)
        XCTAssertLessThanOrEqual(coverage, 0.85)
    }

    func testRealScaleC1CoreMLPointMatchesPythonWhenPresent() async throws {
        let repository = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let model = repository.appendingPathComponent(
            "ml/runs/factored/deployment/compiled/SeeCalScaleC1.mlmodelc",
            isDirectory: true
        )
        let evaluation = repository.appendingPathComponent(
            "ml/runs/factored/scale-v2-probe-c1-fpb-center/" +
                "eval-nutrition5k-overhead-calibrated.json"
        )
        guard FileManager.default.fileExists(atPath: model.path),
              FileManager.default.fileExists(atPath: evaluation.path)
        else {
            throw XCTSkip("Local SCALE C1 artifacts are gitignored")
        }
        let data = try Data(contentsOf: evaluation)
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let paired = try XCTUnwrap(json["paired"] as? [[String: Any]])
        let predictor = try CoreMLScalePredictor(
            modelPath: model.path,
            calibrationMarginGrams: 129.55438232421875
        )
        var relativeErrors = [Double]()
        var absoluteTargetErrors = [Double]()
        for row in paired {
            let groupID = try XCTUnwrap(row["group_id"] as? String)
            let image = repository.appendingPathComponent(
                "ml/dataset_clean/\(groupID)/overhead.jpg"
            )
            let actual = try await predictor.predict(imagePath: image.path)
            let expected = try XCTUnwrap(row["p50_g"] as? Double)
            let target = try XCTUnwrap(row["target_mass_g"] as? Double)
            relativeErrors.append(
                abs(actual.massGrams.estimate - expected) / max(expected, 1)
            )
            absoluteTargetErrors.append(abs(actual.massGrams.estimate - target))
        }
        let meanRelativeDrift =
            relativeErrors.reduce(0, +) / Double(relativeErrors.count)
        let coreMLMAE =
            absoluteTargetErrors.reduce(0, +) / Double(absoluteTargetErrors.count)
        print(
            "[FactoredNutritionPipelineTests] SCALE C1 Core ML: " +
                "mean P50 relative drift=\(meanRelativeDrift), MAE=\(coreMLMAE)"
        )
        XCTAssertLessThan(meanRelativeDrift, 0.10)
        XCTAssertLessThan(coreMLMAE, 44.13)
    }

    func testRealPrunedNutritionDatabaseWhenPresent() async throws {
        let repository = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let database = repository.appendingPathComponent(
            "ml/datasets/fdc/seecal-nutrition.sqlite"
        )
        guard FileManager.default.fileExists(atPath: database.path) else {
            throw XCTSkip("Local FDC database is gitignored")
        }
        let resolver = try SQLiteNutritionResolver(databaseURL: database)
        let cucumber = try await resolver.resolve(name: "cucumber")
        XCTAssertNotEqual(cucumber.rung, .unresolved)
        let profile = try XCTUnwrap(cucumber.profile)
        XCTAssertGreaterThan(profile.carbsPer100g, 0)
        XCTAssertEqual(
            4 * profile.proteinPer100g
                + 9 * profile.fatPer100g
                + 4 * profile.carbsPer100g,
            profile.kcalPer100g,
            accuracy: 0.0001
        )
        let expectedResolutions: [String: (NutritionResolutionRung, String)] = [
            "chicken breast": (.exactAlias, "Poultry Products"),
            "carrot": (.exactAlias, "Vegetables and Vegetable Products"),
            "steak": (.exactAlias, "Beef Products"),
            "shrimp nigiri": (.categoryDefault, "Shellfish"),
        ]
        for (name, expected) in expectedResolutions {
            let resolution = try await resolver.resolve(name: name)
            XCTAssertEqual(resolution.rung, expected.0)
            XCTAssertEqual(resolution.profile?.category, expected.1)
        }
        let chickenAlternatives = await resolver.candidates(
            for: "chicken breast",
            limit: 5
        )
        XCTAssertEqual(chickenAlternatives.count, 5)
        XCTAssertEqual(
            chickenAlternatives.prefix(5).map(\.displayName),
            ["Turkey", "Tofu", "Salmon", "Egg", "Paneer"]
        )
        XCTAssertTrue(chickenAlternatives.allSatisfy { $0.profile.kcalPer100g > 0 })
    }
}
