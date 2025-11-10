#include "jni-utils.h"
#include <cstring>

namespace rnbridge {

std::string sanitize_utf8_for_jni(const char* text) {
    if (!text) return "";

    std::string result;
    result.reserve(strlen(text));

    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(text);
    size_t i = 0;

    while (bytes[i] != 0) {
        unsigned char c = bytes[i];

        if (c <= 0x7F) {
            result += static_cast<char>(c);
            i++;
        } else if ((c & 0xE0) == 0xC0 && (bytes[i+1] & 0xC0) == 0x80) {
            result += static_cast<char>(bytes[i]);
            result += static_cast<char>(bytes[i+1]);
            i += 2;
        } else if ((c & 0xF0) == 0xE0 &&
                   (bytes[i+1] & 0xC0) == 0x80 &&
                   (bytes[i+2] & 0xC0) == 0x80) {
            result += static_cast<char>(bytes[i]);
            result += static_cast<char>(bytes[i+1]);
            result += static_cast<char>(bytes[i+2]);
            i += 3;
        } else if ((c & 0xF8) == 0xF0 &&
                   (bytes[i+1] & 0xC0) == 0x80 &&
                   (bytes[i+2] & 0xC0) == 0x80 &&
                   (bytes[i+3] & 0xC0) == 0x80) {
            result += static_cast<char>(bytes[i]);
            result += static_cast<char>(bytes[i+1]);
            result += static_cast<char>(bytes[i+2]);
            result += static_cast<char>(bytes[i+3]);
            i += 4;
        } else {
            result += '?';
            i++;
        }
    }

    return result;
}

} // namespace rnbridge
