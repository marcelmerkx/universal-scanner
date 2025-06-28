# Contributing to Universal Scanner

We love contributions! Whether it's bug fixes, new features, or documentation improvements, we appreciate your help in making Universal Scanner better.

## Code of Conduct

We want this community to be friendly and respectful. Please be considerate and constructive in all interactions.

## Development Setup

### Prerequisites

- Node.js 16+
- Yarn (preferred) or npm
- For iOS: macOS with Xcode 14+
- For Android: Android Studio with Android SDK 23+
- CMake 3.18+ (for C++ compilation)
- Python 3.8+ (for training scripts)

### Initial Setup

1. Clone the repository:
```sh
git clone https://github.com/yourusername/react-native-universal-scanner
cd react-native-universal-scanner
```

2. Install dependencies:
```sh
yarn install
```

3. For iOS development:
```sh
cd example/ios && pod install
```

## Development Workflow

While developing, you can run the [example app](/example/) to test your changes. TypeScript/JavaScript changes are reflected immediately, but native code changes require rebuilding.

To start the packager:

```sh
yarn example start
```

To run the example app on Android:

```sh
yarn example android
```

To run the example app on iOS:

```sh
yarn example ios
```

Make sure your code passes TypeScript and ESLint. Run the following to verify:

```sh
yarn typecheck
yarn lint
```

To fix formatting errors, run the following:

```sh
yarn lint --fix
```

Remember to add tests for your change if possible. Run the unit tests by:

```sh
yarn test
```

### Native Development

#### iOS (Objective-C++/Swift)
Open `example/ios/UniversalScannerExample.xcworkspace` in Xcode and find the source files at:
- `Pods > Development Pods > react-native-universal-scanner` for the plugin code
- `ios/` directory for iOS-specific implementations

#### Android (Kotlin/Java)
Open `example/android` in Android Studio and find the source files at:
- `react-native-universal-scanner` module for the plugin code
- `android/src/main/java/com/universalscanner/` for Android-specific implementations
- `android/src/main/cpp/` for C++ JNI code

#### C++ Core
The shared C++ code is located in:
- `cpp/` directory for cross-platform scanner engine
- Platform-specific bridges in `android/src/main/cpp/` and `ios/cpp/`


### Commit message convention

We follow the [conventional commits specification](https://www.conventionalcommits.org/en) for our commit messages:

- `fix`: bug fixes, e.g. fix crash due to deprecated method.
- `feat`: new features, e.g. add new method to the module.
- `refactor`: code refactor, e.g. migrate from class components to hooks.
- `docs`: changes into documentation, e.g. add usage example for the module.
- `test`: adding or updating tests, e.g. add integration tests using detox.
- `chore`: tooling changes, e.g. change CI config.

Our pre-commit hooks verify that your commit message matches this format when committing.

### Linting and tests

[ESLint](https://eslint.org/), [Prettier](https://prettier.io/), [TypeScript](https://www.typescriptlang.org/)

We use [TypeScript](https://www.typescriptlang.org/) for type checking, [ESLint](https://eslint.org/) with [Prettier](https://prettier.io/) for linting and formatting the code, and [Jest](https://jestjs.io/) for testing.

Our pre-commit hooks verify that the linter and tests pass when committing.

### Publishing to npm

We use [release-it](https://github.com/release-it/release-it) to make it easier to publish new versions. It handles common tasks like bumping version based on semver, creating tags and releases etc.

To publish new versions, run the following:

```sh
yarn release
```

### Scripts

The `package.json` file contains various scripts for common tasks:

- `yarn bootstrap`: setup project by installing all dependencies and pods.
- `yarn typecheck`: type-check files with TypeScript.
- `yarn lint`: lint files with ESLint.
- `yarn test`: run unit tests with Jest.
- `yarn example start`: start the Metro server for the example app.
- `yarn example android`: run the example app on Android.
- `yarn example ios`: run the example app on iOS.

### Key Areas for Contribution

1. **New Code Type Support**: Add detection for new code types (e.g., Data Matrix, PDF417)
2. **Performance Optimization**: Improve frame processing speed or memory usage
3. **ML Model Updates**: Train and integrate better YOLO models
4. **Platform Features**: Add platform-specific optimizations
5. **Documentation**: Improve guides, API docs, or examples
6. **Testing**: Add unit tests, integration tests, or test utilities
7. **Bug Fixes**: Fix issues reported by the community

### Project Structure

```
react-native-universal-scanner/
├── src/                    # TypeScript source (API, types, hooks)
├── cpp/                    # Shared C++ scanner engine
├── android/               # Android native implementation
│   ├── src/main/java/     # Kotlin/Java code
│   └── src/main/cpp/      # JNI bridge code
├── ios/                   # iOS native implementation
├── example/               # Demo application
├── detection/             # ML training scripts and datasets
└── docs/                  # Documentation
```

### Sending a Pull Request

> **Working on your first pull request?** You can learn how from this _free_ series: [How to Contribute to an Open Source Project on GitHub](https://app.egghead.io/playlists/how-to-contribute-to-an-open-source-project-on-github).

When you're sending a pull request:

- Prefer small pull requests focused on one change
- Verify that linters and tests are passing
- Update documentation for API changes
- Add tests for new functionality
- Follow the pull request template
- For major changes, discuss first by opening an issue

### Pull Request Guidelines

1. **Title**: Use conventional commit format (e.g., `feat: add Data Matrix support`)
2. **Description**: Clearly explain what changed and why
3. **Testing**: Describe how you tested the changes
4. **Screenshots**: Include screenshots for UI changes
5. **Breaking Changes**: Clearly mark any breaking changes
