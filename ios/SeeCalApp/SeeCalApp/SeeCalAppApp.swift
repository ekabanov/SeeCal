//
//  SeeCalAppApp.swift
//  SeeCalApp
//
//  Created by Jevgeni Kabanov on 09.03.2026.
//

import SwiftUI
import SeeCalApp
import SeeCalInference
import Darwin

@main
struct SeeCaliOSApp: App {
    init() {
        setbuf(stdout, nil)
        setbuf(stderr, nil)
        print("[SeeCal][App] launch")
    }

    var body: some Scene {
        WindowGroup {
            ProductionRootView(
                config: QwenRuntimeConfig(
                    modelPath: ModelAssetResolver.resolveModelPath(),
                    adapterPath: ModelAssetResolver.resolveAdapterPath(),
                    runtimePolicy: .mlxOnly,
                    maxOutputTokens: 1024,
                    temperature: 0.1,
                    timeoutSeconds: 180,
                    maxAttemptsPerRuntime: 1
                )
            )
        }
    }
}
