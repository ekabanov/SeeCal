import SwiftUI
import SeeCalDomain
import SeeCalPersistence

#if canImport(PhotosUI)
import PhotosUI
#endif

public struct RootView: View {
    @StateObject private var viewModel: AppViewModel
    @State private var editingEntry: MealLogEntry?
    @State private var draft: MealEditDraft?
    @State private var isEditingGoal = false
    @State private var goalDraft = GoalEditDraft(target: .defaultTarget)
    @State private var isShowingOnboarding = false
    @State private var onboardingDraft = OnboardingDraft()

    @State private var selectedMealType: MealType = .lunch
    @State private var userHint = ""

    #if canImport(PhotosUI)
    @State private var selectedPhotoItem: PhotosPickerItem?
    #endif

    public init(viewModel: AppViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    public var body: some View {
        NavigationStack {
            List {
                Section("Add Meal") {
                    Picker("Meal Type", selection: $selectedMealType) {
                        ForEach(MealType.allCases, id: \.self) { type in
                            Text(type.rawValue.capitalized).tag(type)
                        }
                    }

                    TextField("Optional hint (e.g. chicken rice bowl)", text: $userHint)

                    #if canImport(PhotosUI)
                    PhotosPicker(
                        selection: $selectedPhotoItem,
                        matching: .images,
                        photoLibrary: .shared()
                    ) {
                        Label("Choose Food Photo", systemImage: "camera.viewfinder")
                    }
                    .disabled(viewModel.isScanning)
                    #else
                    Text("Photo picker is available in iOS app targets.")
                        .foregroundStyle(.secondary)
                    #endif

                    if viewModel.isScanning {
                        HStack {
                            ProgressView()
                            Text("Analyzing photo…")
                        }
                    }

                    if let seconds = viewModel.lastInferenceSeconds {
                        Text(String(format: "Last inference: %.1fs", seconds))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Daily Target") {
                    let consumed = viewModel.consumedToday
                    let remaining = viewModel.remainingToday
                    let target = viewModel.dailyTarget
                    Text("Target: \(Int(target.calories)) kcal")
                        .fontWeight(.medium)
                    Text("Consumed: \(Int(consumed.calories)) kcal")
                    Text("Remaining: \(Int(remaining.calories)) kcal")
                        .fontWeight(.semibold)
                    Text("P \(Int(remaining.proteinGrams))g • F \(Int(remaining.fatGrams))g • C \(Int(remaining.carbsGrams))g left")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button("Edit Goal") {
                        goalDraft = GoalEditDraft(target: target)
                        isEditingGoal = true
                    }
                }

                if let profile = viewModel.userProfile {
                    Section("Profile") {
                        Text("Sex: \(profile.biologicalSex.rawValue.capitalized)")
                        Text("Age: \(profile.ageYears)")
                        Text("Height: \(Int(profile.heightCm)) cm")
                        Text(String(format: "Weight: %.1f kg", profile.weightKg))
                        Text("Activity: \(activityTitle(profile.activityLevel))")
                        Text("Goal Pace: \(goalPaceTitle(profile.goalPace))")
                        Button("Edit Profile") {
                            onboardingDraft = OnboardingDraft(profile: profile)
                            isShowingOnboarding = true
                        }
                    }
                }

                Section("Today") {
                    if viewModel.mealEntries.isEmpty {
                        Text("No meals logged yet")
                            .foregroundStyle(.secondary)
                    }

                    ForEach(viewModel.mealEntries) { entry in
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(entry.mealType.rawValue.capitalized)
                                    .font(.headline)
                                Text("\(Int(entry.scanResult.totalCalories)) kcal")
                                    .font(.subheadline)
                                Text("P \(Int(entry.scanResult.proteinGrams))g • F \(Int(entry.scanResult.fatGrams))g • C \(Int(entry.scanResult.carbsGrams))g")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Edit") {
                                editingEntry = entry
                                draft = MealEditDraft(entry: entry)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    .onDelete { indexSet in
                        // Capture the entries to delete up front: `deleteMeal` refetches
                        // and re-sorts `mealEntries` after each mutation, so re-reading
                        // `viewModel.mealEntries[index]` inside the loop would apply
                        // stale indices to a shifted array.
                        let entriesToDelete = indexSet.map { viewModel.mealEntries[$0] }
                        Task {
                            for entry in entriesToDelete {
                                await viewModel.deleteMeal(id: entry.id)
                            }
                        }
                    }
                }

                Section("7-Day Progress") {
                    ForEach(viewModel.weeklyProgress, id: \.dayStart) { point in
                        HStack {
                            Text(point.dayStart, style: .date)
                            Spacer()
                            Text("\(Int(point.calories)) kcal")
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Section("Weight") {
                    Button("Add Sample Weight (kg)") {
                        Task { await viewModel.addWeightEntry(kg: 78.4) }
                    }

                    ForEach(viewModel.weightEntries) { entry in
                        HStack {
                            Text(entry.date, style: .date)
                            Spacer()
                            Text(String(format: "%.1f kg", entry.weightKg))
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Section("8-Week Weight Trend") {
                    ForEach(viewModel.weeklyWeightTrend, id: \.weekStart) { point in
                        HStack {
                            Text(point.weekStart, style: .date)
                            Spacer()
                            Text(point.averageWeightKg > 0 ? String(format: "%.1f kg", point.averageWeightKg) : "-")
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("SeeCal")
            .task {
                await viewModel.loadEntries()
                if viewModel.requiresOnboarding {
                    onboardingDraft = OnboardingDraft()
                    isShowingOnboarding = true
                }
            }
            #if canImport(PhotosUI)
            .onChange(of: selectedPhotoItem) { _, newItem in
                guard let newItem else { return }
                Task {
                    await processSelectedPhoto(newItem)
                }
            }
            #endif
            .alert("Scan Error", isPresented: Binding(
                get: { viewModel.lastError != nil },
                set: { newValue in
                    if !newValue { viewModel.lastError = nil }
                }
            )) {
                Button("OK", role: .cancel) { viewModel.lastError = nil }
            } message: {
                Text(viewModel.lastError ?? "Unknown error")
            }
            .sheet(item: $editingEntry) { entry in
                if let draft {
                    MealEditSheet(
                        draft: draft,
                        onCancel: {
                            editingEntry = nil
                            self.draft = nil
                        },
                        onSave: { savedDraft in
                            Task {
                                do {
                                    let updatedResult = try savedDraft.toFoodScanResult(basedOn: entry.scanResult)
                                    await viewModel.updateMeal(entry, with: updatedResult)
                                } catch {
                                    viewModel.lastError = error.localizedDescription
                                }
                                editingEntry = nil
                                self.draft = nil
                            }
                        }
                    )
                }
            }
            .sheet(isPresented: $isEditingGoal) {
                GoalEditSheet(
                    draft: goalDraft,
                    onCancel: { isEditingGoal = false },
                    onSave: { savedDraft in
                        Task {
                            do {
                                let newTarget = try savedDraft.toDailyTarget()
                                await viewModel.updateDailyTarget(newTarget)
                            } catch {
                                viewModel.lastError = error.localizedDescription
                            }
                            isEditingGoal = false
                        }
                    }
                )
            }
            .sheet(isPresented: $isShowingOnboarding) {
                OnboardingSheet(
                    draft: onboardingDraft,
                    isFirstRun: viewModel.requiresOnboarding,
                    onCancel: {
                        if !viewModel.requiresOnboarding {
                            isShowingOnboarding = false
                        }
                    },
                    onSave: { savedDraft in
                        Task {
                            do {
                                let profile = try savedDraft.toUserProfile()
                                await viewModel.completeOnboarding(with: profile)
                                onboardingDraft = savedDraft
                                isShowingOnboarding = false
                            } catch {
                                viewModel.lastError = error.localizedDescription
                            }
                        }
                    }
                )
                .interactiveDismissDisabled(viewModel.requiresOnboarding)
            }
        }
    }

    private func activityTitle(_ level: ActivityLevel) -> String {
        switch level {
        case .sedentary:
            return "Sedentary"
        case .lightlyActive:
            return "Lightly Active"
        case .moderatelyActive:
            return "Moderately Active"
        case .veryActive:
            return "Very Active"
        }
    }

    private func goalPaceTitle(_ pace: GoalPace) -> String {
        switch pace {
        case .loseSlow:
            return "Lose 0.25 kg/week"
        case .loseModerate:
            return "Lose 0.5 kg/week"
        case .maintain:
            return "Maintain"
        case .gainSlow:
            return "Gain 0.25 kg/week"
        }
    }

    #if canImport(PhotosUI)
    private func processSelectedPhoto(_ item: PhotosPickerItem) async {
        do {
            guard let data = try await item.loadTransferable(type: Data.self), !data.isEmpty else {
                throw NSError(domain: "RootView", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not load selected image"])
            }
            let imagePath = try persistMealImageData(data)
            let hint = userHint.trimmingCharacters(in: .whitespacesAndNewlines)
            await viewModel.addMealPhoto(
                imagePath: imagePath,
                mealType: selectedMealType,
                userHint: hint.isEmpty ? nil : hint
            )
            selectedPhotoItem = nil
        } catch {
            viewModel.lastError = error.localizedDescription
        }
    }

    private func persistMealImageData(_ data: Data) throws -> String {
        let directory = FileBackedStoreLocations.imagesDirectory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("seecal_\(UUID().uuidString).jpg")
        do {
            let processed = try InferenceImagePreprocessor.downsampledJPEG(from: data)
            print("[SeeCal][ImagePreprocess] bytes original=\(data.count) processed=\(processed.count)")
            try processed.write(to: url, options: .atomic)
        } catch {
            print("[SeeCal][ImagePreprocess] failed, using original image data. error=\(error)")
            try data.write(to: url, options: .atomic)
        }
        return url.path
    }
    #endif
}

private struct MealEditSheet: View {
    @State var draft: MealEditDraft
    let onCancel: () -> Void
    let onSave: (MealEditDraft) -> Void

    var body: some View {
        NavigationStack {
            Form {
                Section("Adjust Nutrition") {
                    TextField("Calories", text: $draft.caloriesText)
                    TextField("Protein (g)", text: $draft.proteinText)
                    TextField("Fat (g)", text: $draft.fatText)
                    TextField("Carbs (g)", text: $draft.carbsText)
                }
            }
            .navigationTitle("Edit Meal")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel", action: onCancel)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { onSave(draft) }
                }
            }
        }
    }
}

private struct OnboardingSheet: View {
    @State var draft: OnboardingDraft
    let isFirstRun: Bool
    let onCancel: () -> Void
    let onSave: (OnboardingDraft) -> Void

    var body: some View {
        NavigationStack {
            Form {
                Section("About You") {
                    Picker("Sex", selection: $draft.biologicalSex) {
                        Text("Male").tag(BiologicalSex.male)
                        Text("Female").tag(BiologicalSex.female)
                    }
                    TextField("Age (years)", text: $draft.ageYearsText)
                    TextField("Height (cm)", text: $draft.heightCmText)
                    TextField("Weight (kg)", text: $draft.weightKgText)
                }

                Section("Lifestyle") {
                    Picker("Activity", selection: $draft.activityLevel) {
                        Text("Sedentary").tag(ActivityLevel.sedentary)
                        Text("Lightly Active").tag(ActivityLevel.lightlyActive)
                        Text("Moderately Active").tag(ActivityLevel.moderatelyActive)
                        Text("Very Active").tag(ActivityLevel.veryActive)
                    }
                    Picker("Goal Pace", selection: $draft.goalPace) {
                        Text("Lose 0.25 kg/week").tag(GoalPace.loseSlow)
                        Text("Lose 0.5 kg/week").tag(GoalPace.loseModerate)
                        Text("Maintain").tag(GoalPace.maintain)
                        Text("Gain 0.25 kg/week").tag(GoalPace.gainSlow)
                    }
                }
            }
            .navigationTitle("Setup")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    if !isFirstRun {
                        Button("Cancel", action: onCancel)
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { onSave(draft) }
                }
            }
        }
    }
}

private struct GoalEditSheet: View {
    @State var draft: GoalEditDraft
    let onCancel: () -> Void
    let onSave: (GoalEditDraft) -> Void

    var body: some View {
        NavigationStack {
            Form {
                Section("Daily Goal") {
                    TextField("Calories", text: $draft.caloriesText)
                    TextField("Protein (g)", text: $draft.proteinText)
                    TextField("Fat (g)", text: $draft.fatText)
                    TextField("Carbs (g)", text: $draft.carbsText)
                }
            }
            .navigationTitle("Edit Goal")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel", action: onCancel)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { onSave(draft) }
                }
            }
        }
    }
}
