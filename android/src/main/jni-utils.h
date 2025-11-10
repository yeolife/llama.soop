#ifndef JNI_UTILS_H
#define JNI_UTILS_H

#include <jni.h>
#include <string>

namespace rnbridge {
std::string sanitize_utf8_for_jni(const char* text);
}

namespace jnihelpers {
inline jclass findClass(JNIEnv *env, const char *name) {
    return env->FindClass(name);
}

inline bool instanceOf(JNIEnv *env, jobject obj, const char *className) {
    if (!obj) return false;
    jclass cls = env->FindClass(className);
    bool result = env->IsInstanceOf(obj, cls);
    env->DeleteLocalRef(cls);
    return result;
}

inline jobject mapGet(JNIEnv *env, jobject map, const char *key) {
    if (!map) return nullptr;
    jclass mapClass = env->GetObjectClass(map);
    jmethodID getMethod = env->GetMethodID(mapClass, "get", "(Ljava/lang/Object;)Ljava/lang/Object;");
    env->DeleteLocalRef(mapClass);
    jstring jKey = env->NewStringUTF(key);
    jobject value = env->CallObjectMethod(map, getMethod, jKey);
    env->DeleteLocalRef(jKey);
    return value;
}

inline bool mapContainsKey(JNIEnv *env, jobject map, const char *key) {
    if (!map) return false;
    jclass mapClass = env->GetObjectClass(map);
    jmethodID containsKey = env->GetMethodID(mapClass, "containsKey", "(Ljava/lang/Object;)Z");
    jstring jKey = env->NewStringUTF(key);
    jboolean result = env->CallBooleanMethod(map, containsKey, jKey);
    env->DeleteLocalRef(jKey);
    env->DeleteLocalRef(mapClass);
    return result == JNI_TRUE;
}
}

namespace maputils {

inline jobject createWriteableMap(JNIEnv *env) {
    jclass hashMapClass = env->FindClass("java/util/HashMap");
    jmethodID init = env->GetMethodID(hashMapClass, "<init>", "()V");
    jobject map = env->NewObject(hashMapClass, init);
    env->DeleteLocalRef(hashMapClass);
    return map;
}

inline void putString(JNIEnv *env, jobject map, const char *key, const char *value) {
    if (!map) return;
    jclass mapClass = env->GetObjectClass(map);
    jmethodID putMethod = env->GetMethodID(mapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jstring jKey = env->NewStringUTF(key);
    std::string sanitized = rnbridge::sanitize_utf8_for_jni(value);
    jstring jValue = env->NewStringUTF(sanitized.c_str());
    env->CallObjectMethod(map, putMethod, jKey, jValue);
    env->DeleteLocalRef(jKey);
    env->DeleteLocalRef(jValue);
    env->DeleteLocalRef(mapClass);
}

inline void putInt(JNIEnv *env, jobject map, const char *key, int value) {
    if (!map) return;
    jclass integerClass = env->FindClass("java/lang/Integer");
    jmethodID valueOf = env->GetStaticMethodID(integerClass, "valueOf", "(I)Ljava/lang/Integer;");
    jobject integerObj = env->CallStaticObjectMethod(integerClass, valueOf, value);
    env->DeleteLocalRef(integerClass);

    jclass mapClass = env->GetObjectClass(map);
    jmethodID putMethod = env->GetMethodID(mapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jstring jKey = env->NewStringUTF(key);
    env->CallObjectMethod(map, putMethod, jKey, integerObj);
    env->DeleteLocalRef(jKey);
    env->DeleteLocalRef(integerObj);
    env->DeleteLocalRef(mapClass);
}

inline void putDouble(JNIEnv *env, jobject map, const char *key, double value) {
    if (!map) return;
    jclass doubleClass = env->FindClass("java/lang/Double");
    jmethodID valueOf = env->GetStaticMethodID(doubleClass, "valueOf", "(D)Ljava/lang/Double;");
    jobject doubleObj = env->CallStaticObjectMethod(doubleClass, valueOf, value);
    env->DeleteLocalRef(doubleClass);

    jclass mapClass = env->GetObjectClass(map);
    jmethodID putMethod = env->GetMethodID(mapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jstring jKey = env->NewStringUTF(key);
    env->CallObjectMethod(map, putMethod, jKey, doubleObj);
    env->DeleteLocalRef(jKey);
    env->DeleteLocalRef(doubleObj);
    env->DeleteLocalRef(mapClass);
}

inline void putBoolean(JNIEnv *env, jobject map, const char *key, bool value) {
    if (!map) return;
    jclass booleanClass = env->FindClass("java/lang/Boolean");
    jmethodID valueOf = env->GetStaticMethodID(booleanClass, "valueOf", "(Z)Ljava/lang/Boolean;");
    jobject boolObj = env->CallStaticObjectMethod(booleanClass, valueOf, value ? JNI_TRUE : JNI_FALSE);
    env->DeleteLocalRef(booleanClass);

    jclass mapClass = env->GetObjectClass(map);
    jmethodID putMethod = env->GetMethodID(mapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jstring jKey = env->NewStringUTF(key);
    env->CallObjectMethod(map, putMethod, jKey, boolObj);
    env->DeleteLocalRef(jKey);
    env->DeleteLocalRef(boolObj);
    env->DeleteLocalRef(mapClass);
}

inline void putMap(JNIEnv *env, jobject map, const char *key, jobject value) {
    if (!map) return;
    jclass mapClass = env->GetObjectClass(map);
    jmethodID putMethod = env->GetMethodID(mapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jstring jKey = env->NewStringUTF(key);
    env->CallObjectMethod(map, putMethod, jKey, value);
    env->DeleteLocalRef(jKey);
    env->DeleteLocalRef(mapClass);
}

inline void putArray(JNIEnv *env, jobject map, const char *key, jobject value) {
    putMap(env, map, key, value);
}

} // namespace maputils

namespace listutils {

inline jobject createWritableArray(JNIEnv *env) {
    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID init = env->GetMethodID(arrayListClass, "<init>", "()V");
    jobject array = env->NewObject(arrayListClass, init);
    env->DeleteLocalRef(arrayListClass);
    return array;
}

inline void pushInt(JNIEnv *env, jobject arr, int value) {
    if (!arr) return;
    jclass arrayClass = env->GetObjectClass(arr);
    jmethodID addMethod = env->GetMethodID(arrayClass, "add", "(Ljava/lang/Object;)Z");
    jclass integerClass = env->FindClass("java/lang/Integer");
    jmethodID valueOf = env->GetStaticMethodID(integerClass, "valueOf", "(I)Ljava/lang/Integer;");
    jobject integerObj = env->CallStaticObjectMethod(integerClass, valueOf, value);
    env->CallBooleanMethod(arr, addMethod, integerObj);
    env->DeleteLocalRef(integerObj);
    env->DeleteLocalRef(integerClass);
    env->DeleteLocalRef(arrayClass);
}

inline void pushDouble(JNIEnv *env, jobject arr, double value) {
    if (!arr) return;
    jclass arrayClass = env->GetObjectClass(arr);
    jmethodID addMethod = env->GetMethodID(arrayClass, "add", "(Ljava/lang/Object;)Z");
    jclass doubleClass = env->FindClass("java/lang/Double");
    jmethodID valueOf = env->GetStaticMethodID(doubleClass, "valueOf", "(D)Ljava/lang/Double;");
    jobject doubleObj = env->CallStaticObjectMethod(doubleClass, valueOf, value);
    env->CallBooleanMethod(arr, addMethod, doubleObj);
    env->DeleteLocalRef(doubleObj);
    env->DeleteLocalRef(doubleClass);
    env->DeleteLocalRef(arrayClass);
}

inline void pushString(JNIEnv *env, jobject arr, const char *value) {
    if (!arr) return;
    jclass arrayClass = env->GetObjectClass(arr);
    jmethodID addMethod = env->GetMethodID(arrayClass, "add", "(Ljava/lang/Object;)Z");
    std::string sanitized = rnbridge::sanitize_utf8_for_jni(value);
    jstring jValue = env->NewStringUTF(sanitized.c_str());
    env->CallBooleanMethod(arr, addMethod, jValue);
    env->DeleteLocalRef(jValue);
    env->DeleteLocalRef(arrayClass);
}

inline void pushMap(JNIEnv *env, jobject arr, jobject value) {
    if (!arr) return;
    jclass arrayClass = env->GetObjectClass(arr);
    jmethodID addMethod = env->GetMethodID(arrayClass, "add", "(Ljava/lang/Object;)Z");
    env->CallBooleanMethod(arr, addMethod, value);
    env->DeleteLocalRef(arrayClass);
}

} // namespace listutils

namespace listreader {

inline int size(JNIEnv *env, jobject readableArray) {
    if (!readableArray) return 0;
    jclass arrayClass = env->GetObjectClass(readableArray);
    jmethodID sizeMethod = env->GetMethodID(arrayClass, "size", "()I");
    jint result = env->CallIntMethod(readableArray, sizeMethod);
    env->DeleteLocalRef(arrayClass);
    return result;
}

inline jobject getMap(JNIEnv *env, jobject readableArray, int index) {
    if (!readableArray) return nullptr;
    jclass arrayClass = env->GetObjectClass(readableArray);
    jmethodID getMethod = env->GetMethodID(arrayClass, "get", "(I)Ljava/lang/Object;");
    jobject value = env->CallObjectMethod(readableArray, getMethod, index);
    env->DeleteLocalRef(arrayClass);
    return value;
}

inline jstring getString(JNIEnv *env, jobject readableArray, int index) {
    jobject value = getMap(env, readableArray, index);
    if (!value) return nullptr;
    if (jnihelpers::instanceOf(env, value, "java/lang/String")) {
        return (jstring) value;
    }
    jclass objClass = env->GetObjectClass(value);
    jmethodID toString = env->GetMethodID(objClass, "toString", "()Ljava/lang/String;");
    jstring str = (jstring) env->CallObjectMethod(value, toString);
    env->DeleteLocalRef(objClass);
    env->DeleteLocalRef(value);
    return str;
}

} // namespace listreader

namespace mapreader {

inline bool hasKey(JNIEnv *env, jobject readableMap, const char *key) {
    return jnihelpers::mapContainsKey(env, readableMap, key);
}

inline int getInt(JNIEnv *env, jobject readableMap, const char *key, jint defaultValue) {
    if (!mapreader::hasKey(env, readableMap, key)) return defaultValue;
    jobject value = jnihelpers::mapGet(env, readableMap, key);
    if (!value) return defaultValue;
    jclass numberClass = env->FindClass("java/lang/Number");
    int result = defaultValue;
    if (env->IsInstanceOf(value, numberClass)) {
        jmethodID intValue = env->GetMethodID(numberClass, "intValue", "()I");
        result = env->CallIntMethod(value, intValue);
    }
    env->DeleteLocalRef(numberClass);
    env->DeleteLocalRef(value);
    return result;
}

inline bool getBool(JNIEnv *env, jobject readableMap, const char *key, jboolean defaultValue) {
    if (!hasKey(env, readableMap, key)) return defaultValue;
    jobject value = jnihelpers::mapGet(env, readableMap, key);
    if (!value) return defaultValue;
    jclass booleanClass = env->FindClass("java/lang/Boolean");
    bool result = defaultValue;
    if (env->IsInstanceOf(value, booleanClass)) {
        jmethodID booleanValue = env->GetMethodID(booleanClass, "booleanValue", "()Z");
        result = env->CallBooleanMethod(value, booleanValue);
    }
    env->DeleteLocalRef(booleanClass);
    env->DeleteLocalRef(value);
    return result;
}

inline long getLong(JNIEnv *env, jobject readableMap, const char *key, jlong defaultValue) {
    if (!hasKey(env, readableMap, key)) return defaultValue;
    jobject value = jnihelpers::mapGet(env, readableMap, key);
    if (!value) return defaultValue;
    jclass numberClass = env->FindClass("java/lang/Number");
    long result = defaultValue;
    if (env->IsInstanceOf(value, numberClass)) {
        jmethodID longValue = env->GetMethodID(numberClass, "longValue", "()J");
        result = env->CallLongMethod(value, longValue);
    }
    env->DeleteLocalRef(numberClass);
    env->DeleteLocalRef(value);
    return result;
}

inline float getFloat(JNIEnv *env, jobject readableMap, const char *key, jfloat defaultValue) {
    if (!hasKey(env, readableMap, key)) return defaultValue;
    jobject value = jnihelpers::mapGet(env, readableMap, key);
    if (!value) return defaultValue;
    jclass numberClass = env->FindClass("java/lang/Number");
    float result = defaultValue;
    if (env->IsInstanceOf(value, numberClass)) {
        jmethodID doubleValue = env->GetMethodID(numberClass, "doubleValue", "()D");
        result = static_cast<float>(env->CallDoubleMethod(value, doubleValue));
    }
    env->DeleteLocalRef(numberClass);
    env->DeleteLocalRef(value);
    return result;
}

inline jstring getString(JNIEnv *env, jobject readableMap, const char *key, jstring defaultValue) {
    if (!hasKey(env, readableMap, key)) return defaultValue;
    jobject value = jnihelpers::mapGet(env, readableMap, key);
    if (!value) return defaultValue;
    if (jnihelpers::instanceOf(env, value, "java/lang/String")) {
        return (jstring) value;
    }
    jclass objClass = env->GetObjectClass(value);
    jmethodID toString = env->GetMethodID(objClass, "toString", "()Ljava/lang/String;");
    jstring str = (jstring) env->CallObjectMethod(value, toString);
    env->DeleteLocalRef(objClass);
    env->DeleteLocalRef(value);
    return str;
}

inline jobject getArray(JNIEnv *env, jobject readableMap, const char *key) {
    jobject value = jnihelpers::mapGet(env, readableMap, key);
    if (!value) return nullptr;
    if (!jnihelpers::instanceOf(env, value, "java/util/List")) {
        env->DeleteLocalRef(value);
        return nullptr;
    }
    return value;
}

} // namespace mapreader

#endif // JNI_UTILS_H
