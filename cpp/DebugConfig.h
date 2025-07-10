#ifndef DEBUG_CONFIG_H
#define DEBUG_CONFIG_H

namespace UniversalScanner {

class DebugConfig {
public:
    static DebugConfig& getInstance() {
        static DebugConfig instance;
        return instance;
    }

    bool isDebugImagesEnabled() const {
        return m_enableDebugImages;
    }

    void setDebugImagesEnabled(bool enabled) {
        m_enableDebugImages = enabled;
    }

private:
    DebugConfig() {
        #ifdef DEBUG
            m_enableDebugImages = true;
        #else
            m_enableDebugImages = false;
        #endif
    }

    bool m_enableDebugImages;

    DebugConfig(const DebugConfig&) = delete;
    DebugConfig& operator=(const DebugConfig&) = delete;
};

}

#endif