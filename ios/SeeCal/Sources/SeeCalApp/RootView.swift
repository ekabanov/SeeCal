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
                    // Computed from userProfile via GoalCalculator (spec §2) — no
                    // longer an independently editable value (GoalEditDraft retired).
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
                }

                if let profile = viewModel.userProfile {
                    Section("Profile") {
                        Text("Sex: \(profile.sex.rawValue.capitalized)")
                        Text("Age: \(GoalCalculator.age(dateOfBirth: profile.dateOfBirth))")
                        Text("Height: \(profile.heightCm) cm")
                        Text(String(format: "Weight: %.1f kg", profile.weightKg))
                        Text("Activity: \(activityTitle(profile.activity))")
                        Text(String(format: "Weekly target: %.1f kg/week", profile.weeklyRateKg))
                        Button("Edit Profile") {
                            onboardingDraft = OnboardingDraft(profile: profile)
                            isShowingOnboarding = true
                        }
                    }
                }

                Section("Today") {
                    // Match the totals above (which use `consumedToday`): only show
                    // entries actually logged today, not the full history.
                    let todaysEntries = viewModel.mealEntries.filter { Calendar.current.isDateInToday($0.createdAt) }

                    if todaysEntries.isEmpty {
                        Text("No meals logged yet")
                            .foregroundStyle(.secondary)
                    }

                    ForEach(todaysEntries) { entry in
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(entry.name)
                                    .font(.headline)
                                Text("\(Int(entry.totals.calories)) kcal")
                                    .font(.subheadline)
                                Text("P \(Int(entry.totals.proteinGrams))g • F \(Int(entry.totals.fatGrams))g • C \(Int(entry.totals.carbsGrams))g")
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
                        // stale indices to a shifted array. Indices here refer to
                        // `todaysEntries` (the filtered, displayed array), not the
                        // unfiltered `viewModel.mealEntries`, so map through that array.
                        let entriesToDelete = indexSet.map { todaysEntries[$0] }
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
                                    let updatedEntry = try savedDraft.committedEntry()
                                    await viewModel.updateMeal(updatedEntry)
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
        case .light:
            return "Light"
        case .moderate:
            return "Moderate"
        case .active:
            return "Active"
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
                // Placeholder layout only — the real result/edit sheet (photo, hero
                // total, macro chips, per-item gram steppers per spec §5) lands in P6.
                // This just exercises MealEditDraft's per-item API well enough to
                // keep RootView compiling.
                Section("Detected Items") {
                    ForEach(draft.items) { item in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(item.name)
                            HStack {
                                Button("-5 g") { draft.stepGrams(itemID: item.id, by: -MealItem.gramStep) }
                                Text("\(Int(item.grams)) g")
                                Button("+5 g") { draft.stepGrams(itemID: item.id, by: MealItem.gramStep) }
                            }
                            Text("\(Int(item.kcal)) kcal")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                Section("Totals") {
                    let totals = draft.totals
                    Text("\(Int(totals.calories)) kcal")
                        .fontWeight(.semibold)
                    Text("P \(Int(totals.proteinGrams))g • F \(Int(totals.fatGrams))g • C \(Int(totals.carbsGrams))g")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle(draft.name)
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
                    Picker("Sex", selection: $draft.sex) {
                        Text("Male").tag(BiologicalSex.male)
                        Text("Female").tag(BiologicalSex.female)
                    }
                    DatePicker("Birthday", selection: $draft.dateOfBirth, displayedComponents: .date)
                    TextField("Height (cm)", text: $draft.heightCmText)
                    TextField("Weight (kg)", text: $draft.weightKgText)
                }

                Section("Lifestyle") {
                    Picker("Activity", selection: $draft.activity) {
                        Text("Sedentary").tag(ActivityLevel.sedentary)
                        Text("Light").tag(ActivityLevel.light)
                        Text("Moderate").tag(ActivityLevel.moderate)
                        Text("Active").tag(ActivityLevel.active)
                    }
                    Stepper(
                        String(format: "Weekly target: %.1f kg/week", draft.weeklyRateKg),
                        value: $draft.weeklyRateKg,
                        in: UserProfile.weeklyRateRange,
                        step: 0.1
                    )
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
