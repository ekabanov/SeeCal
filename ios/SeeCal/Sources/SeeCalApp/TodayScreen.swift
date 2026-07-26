import SwiftUI
import SeeCalDomain
import SeeCalPersistence

#if canImport(PhotosUI)
import PhotosUI
#endif

/// Spec §4 placeholder. The ring card + macro bars + polished meal rows are P4's
/// job; for P3 this parks the existing add-meal / daily-target / meal-list
/// functionality (previously the entirety of `RootView`) under the Today tab.
public struct TodayScreen: View {
    @ObservedObject private var viewModel: AppViewModel

    @State private var editingEntry: MealLogEntry?
    @State private var draft: MealEditDraft?
    @State private var selectedMealType: MealType = .lunch
    @State private var userHint = ""

    #if canImport(PhotosUI)
    @State private var selectedPhotoItem: PhotosPickerItem?
    #endif

    public init(viewModel: AppViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                PageHeader(subtitle: todayDateString, title: "Today")
                    .padding(.top, 8)

                SectionLabel("Add meal")
                Card {
                    addMealContent
                }

                SectionLabel("Daily target")
                Card {
                    dailyTargetContent
                }

                SectionLabel("Meals")
                Card {
                    mealsContent
                }

                PrivacyChip("Analyzed entirely on this iPhone.")
            }
            .padding(.horizontal, 18)
            .padding(.bottom, Theme.screenBottomInset)
        }
        .background(Theme.appBg)
        #if canImport(PhotosUI)
        .onChange(of: selectedPhotoItem) { _, newItem in
            guard let newItem else { return }
            Task { await processSelectedPhoto(newItem) }
        }
        #endif
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
    }

    private var todayDateString: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "EEEE, MMM d"
        return formatter.string(from: Date())
    }

    @ViewBuilder
    private var addMealContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("Meal Type", selection: $selectedMealType) {
                ForEach(MealType.allCases, id: \.self) { type in
                    Text(type.rawValue.capitalized).tag(type)
                }
            }

            TextField("Optional hint (e.g. chicken rice bowl)", text: $userHint)
                .textFieldStyle(.roundedBorder)

            #if canImport(PhotosUI)
            PhotosPicker(
                selection: $selectedPhotoItem,
                matching: .images,
                photoLibrary: .shared()
            ) {
                Label("Choose Food Photo", systemImage: "camera.viewfinder")
            }
            .disabled(viewModel.isScanning)
            .foregroundStyle(Theme.basil)
            #else
            Text("Photo picker is available in iOS app targets.")
                .foregroundStyle(Theme.appInk2)
            #endif

            if viewModel.isScanning {
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Analyzing photo…")
                        .foregroundStyle(Theme.appInk2)
                }
            }

            if let seconds = viewModel.lastInferenceSeconds {
                Text(String(format: "Last inference: %.1fs", seconds))
                    .font(.caption)
                    .foregroundStyle(Theme.appInk2)
            }
        }
        // Scan errors surface through the app-wide error alert owned by `RootView`
        // (bound to the same `viewModel.lastError`) rather than a second local
        // alert here — two alert modifiers racing on one Bool would fight over
        // presentation.
    }

    @ViewBuilder
    private var dailyTargetContent: some View {
        let consumed = viewModel.consumedToday
        let remaining = viewModel.remainingToday
        let target = viewModel.dailyTarget
        VStack(alignment: .leading, spacing: 4) {
            Text("Target: \(Int(target.calories)) kcal")
                .fontWeight(.medium)
            Text("Consumed: \(Int(consumed.calories)) kcal")
            Text("Remaining: \(Int(remaining.calories)) kcal")
                .fontWeight(.semibold)
            Text("P \(Int(remaining.proteinGrams))g • F \(Int(remaining.fatGrams))g • C \(Int(remaining.carbsGrams))g left")
                .font(.caption)
                .foregroundStyle(Theme.appInk2)
        }
        .foregroundStyle(Theme.appInk)
    }

    @ViewBuilder
    private var mealsContent: some View {
        let todaysEntries = viewModel.mealEntries.filter { Calendar.current.isDateInToday($0.createdAt) }

        if todaysEntries.isEmpty {
            Text("No meals logged yet")
                .foregroundStyle(Theme.appInk2)
        } else {
            VStack(spacing: 0) {
                ForEach(Array(todaysEntries.enumerated()), id: \.element.id) { index, entry in
                    if index > 0 {
                        Divider().overlay(Theme.appLine)
                    }
                    mealRow(entry)
                }
            }
        }
    }

    private func mealRow(_ entry: MealLogEntry) -> some View {
        Button {
            editingEntry = entry
            draft = MealEditDraft(entry: entry)
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(entry.name)
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(Theme.appInk)
                    Text("\(Int(entry.totals.calories)) kcal")
                        .font(.system(size: 13.5))
                        .foregroundStyle(Theme.appInk2)
                    Text("P \(Int(entry.totals.proteinGrams))g • F \(Int(entry.totals.fatGrams))g • C \(Int(entry.totals.carbsGrams))g")
                        .font(.caption)
                        .foregroundStyle(Theme.appInk2)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .foregroundStyle(Theme.appInk2)
            }
            .padding(.vertical, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .contextMenu {
            // `.swipeActions` only has an effect inside a `List`; this VStack-based
            // layout (matching the prototype's card-based meal rows, not a system
            // List) uses a context menu instead so deleting an entry stays reachable.
            Button("Delete", role: .destructive) {
                Task { await viewModel.deleteMeal(id: entry.id) }
            }
        }
    }

    #if canImport(PhotosUI)
    private func processSelectedPhoto(_ item: PhotosPickerItem) async {
        do {
            guard let data = try await item.loadTransferable(type: Data.self), !data.isEmpty else {
                throw NSError(domain: "TodayScreen", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not load selected image"])
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

/// Placeholder layout only — the real result/edit sheet (photo, hero total, macro
/// chips, per-item gram steppers per spec §5) lands in P6. This just exercises
/// `MealEditDraft`'s per-item API well enough to keep the Today tab compiling and
/// interactive.
private struct MealEditSheet: View {
    @State var draft: MealEditDraft
    let onCancel: () -> Void
    let onSave: (MealEditDraft) -> Void

    var body: some View {
        NavigationStack {
            Form {
                Section("Detected Items") {
                    ForEach(draft.items) { item in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(item.name)
                            GramStepper(
                                valueText: String(Int(item.grams)),
                                unitText: "g",
                                decrementLabel: "Fewer grams",
                                incrementLabel: "More grams",
                                onDecrement: { draft.stepGrams(itemID: item.id, by: -MealItem.gramStep) },
                                onIncrement: { draft.stepGrams(itemID: item.id, by: MealItem.gramStep) }
                            )
                            Text("\(Int(item.kcal)) kcal")
                                .font(.caption)
                                .foregroundStyle(Theme.appInk2)
                        }
                    }
                }
                Section("Totals") {
                    let totals = draft.totals
                    Text("\(Int(totals.calories)) kcal")
                        .fontWeight(.semibold)
                    Text("P \(Int(totals.proteinGrams))g • F \(Int(totals.fatGrams))g • C \(Int(totals.carbsGrams))g")
                        .font(.caption)
                        .foregroundStyle(Theme.appInk2)
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
