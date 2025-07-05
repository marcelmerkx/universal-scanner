#include "YuvConverter.h"

#ifdef __ANDROID__
#include "AndroidYuvConverter.h"
#endif

#ifdef __APPLE__
#include "IOSYuvConverter.h"
#endif

namespace UniversalScanner {

std::unique_ptr<YuvConverter> YuvConverter::create(
#ifdef __ANDROID__
    JNIEnv* env,
    jobject context
#endif
) {
#ifdef __ANDROID__
    if (env) {
        return std::make_unique<AndroidYuvConverter>(env, context);
    } else {
        return nullptr;
    }
#elif defined(__APPLE__)
    return std::make_unique<IOSYuvConverter>();
#else
    // Fallback for other platforms - could implement a generic converter
    return nullptr;
#endif
}

} // namespace UniversalScanner