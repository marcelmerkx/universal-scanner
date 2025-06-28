# VisionCamera + Worklets Linking Issue - SOLVED

## Problem
VisionCamera failed to build with react-native-worklets-core due to undefined symbol errors during linking. The issue was that the Android prefab system wasn't properly exporting the worklets library for consumption by other CMake projects.

## Root Cause
The react-native-worklets-core library wasn't properly configured for prefab export. While the library built successfully, it wasn't being made available to VisionCamera through Android's prefab system.

## Solution
Instead of trying to fix the complex prefab configuration, we implemented a direct linking approach:

### 1. Modified VisionCamera's CMakeLists.txt
We patched VisionCamera to directly link to the worklets library instead of using prefab's find_package:

```cmake
# Find worklets library directly
set(WORKLETS_DIR "${NODE_MODULES_DIR}/react-native-worklets-core/android")
set(WORKLETS_BUILD_DIR "${WORKLETS_DIR}/build/intermediates/cmake/release/obj/${ANDROID_ABI}")

add_library(rnworklets SHARED IMPORTED)
set_target_properties(rnworklets PROPERTIES IMPORTED_LOCATION "${WORKLETS_BUILD_DIR}/librnworklets.so")
target_include_directories(${PACKAGE_NAME} PRIVATE "${WORKLETS_DIR}/build/headers/rnworklets")

target_link_libraries(
  ${PACKAGE_NAME}
  rnworklets
)
```

### 2. Excluded Duplicate Library
We also had to exclude the worklets library from VisionCamera's packaging to avoid duplicate library errors:

```gradle
packagingOptions {
  excludes = [
    // ... other excludes ...
    "**/librnworklets.so"
  ]
}
```

## Patches Created
Two patch files were created using patch-package:

1. **patches/react-native-vision-camera+4.7.0.patch** - Contains both the CMakeLists.txt changes and the packagingOptions exclusion

## How to Apply
The patches are automatically applied via the postinstall script:
```json
"postinstall": "patch-package"
```

## Testing
After applying these patches:
1. Clean the build: `cd android && ./gradlew clean`
2. Build the app: `./gradlew :app:assembleDebug`

The build now completes successfully with frame processors enabled!

## Note
This is a workaround for what appears to be a prefab configuration issue in react-native-worklets-core. The proper long-term fix would be for the worklets library to properly configure its prefab export, but this solution works reliably in the meantime. 