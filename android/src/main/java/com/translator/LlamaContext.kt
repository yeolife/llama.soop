package com.afreecatv.mobile.sdk.aiEdgeLayer.translator

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.ParcelFileDescriptor
import android.util.Log
import androidx.annotation.Keep
import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.FileReader
import java.io.IOException
import java.io.InputStream
import java.util.HashMap
import java.util.regex.Pattern

/**
 * Pure Android wrapper over the native llama JNI bindings.
 *
 * This class mirrors the old React Native Java implementation but only relies on
 * Kotlin/Java standard types (Map/List/ArrayList) so it can be used directly inside
 * Android applications without the RN bridge.
 */
@Keep
class LlamaContext(
    val id: Int,
    private val androidContext: Context?,
    options: InitOptions,
    progressListener: ProgressListener? = null
) {

    companion object {
        private const val TAG = "LlamaContext"

        private val ggufHeader = byteArrayOf(0x47, 0x47, 0x55, 0x46)
        private var loadedLibrary: String = ""

        init {
            Log.d(TAG, "Primary ABI: ${Build.SUPPORTED_ABIS[0]}")

            val cpuFeatures = getCpuFeatures()
            Log.d(TAG, "CPU features: $cpuFeatures")
            val hasFp16 = cpuFeatures.contains("fp16") || cpuFeatures.contains("fphp")
            val hasDotProd = cpuFeatures.contains("dotprod") || cpuFeatures.contains("asimddp")
            val hasI8mm = cpuFeatures.contains("i8mm")
            val hasAdreno = Pattern.compile("(adreno|qcom|qualcomm)")
                .matcher("${Build.HARDWARE} ${Build.MANUFACTURER} ${Build.MODEL}".lowercase())
                .find()

            when {
                isArm64V8a() && hasDotProd && hasI8mm && hasAdreno -> {
                    System.loadLibrary("llama_jni_v8_2_dotprod_i8mm_opencl")
                    loadedLibrary = "llama_jni_v8_2_dotprod_i8mm_opencl"
                }
                isArm64V8a() && hasDotProd && hasI8mm -> {
                    System.loadLibrary("llama_jni_v8_2_dotprod_i8mm")
                    loadedLibrary = "llama_jni_v8_2_dotprod_i8mm"
                }
                isArm64V8a() && hasDotProd -> {
                    System.loadLibrary("llama_jni_v8_2_dotprod")
                    loadedLibrary = "llama_jni_v8_2_dotprod"
                }
                isArm64V8a() && hasI8mm -> {
                    System.loadLibrary("llama_jni_v8_2_i8mm")
                    loadedLibrary = "llama_jni_v8_2_i8mm"
                }
                isArm64V8a() && hasFp16 -> {
                    System.loadLibrary("llama_jni_v8_2")
                    loadedLibrary = "llama_jni_v8_2"
                }
                isArm64V8a() -> {
                    System.loadLibrary("llama_jni_v8")
                    loadedLibrary = "llama_jni_v8"
                }
                isX86_64() -> {
                    System.loadLibrary("llama_jni_x86_64")
                    loadedLibrary = "llama_jni_x86_64"
                }
                else -> Log.w(TAG, "Unsupported ABI, native libraries were not loaded")
            }
        }

        fun toggleNativeLog(enabled: Boolean, logger: ((level: String, text: String) -> Unit)? = null) {
            if (!isArchSupported()) {
                throw IllegalStateException("Only 64-bit architectures are supported")
            }
            if (enabled) {
                setupLog(NativeLogCallback(logger))
            } else {
                unsetLog()
            }
        }

        fun buildInfo(path: String, skip: List<String> = emptyList()): Map<String, Any?> {
            return modelInfo(path, skip.toTypedArray())
        }

        fun backendDevicesInfo(): String = getBackendDevicesInfo()

        fun loadedNativeLibrary(): String = loadedLibrary

        private fun isArm64V8a(): Boolean = Build.SUPPORTED_ABIS.firstOrNull() == "arm64-v8a"
        private fun isX86_64(): Boolean = Build.SUPPORTED_ABIS.firstOrNull() == "x86_64"

        fun isArchSupported(): Boolean = isArm64V8a() || isX86_64()
        fun isArchNotSupported(): Boolean = !isArchSupported()

        fun cpuFeatures(): String = getCpuFeatures()

        private fun getCpuFeatures(): String {
            val file = File("/proc/cpuinfo")
            val builder = StringBuilder()
            return try {
                BufferedReader(FileReader(file)).use { reader ->
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        if (line!!.startsWith("Features")) {
                            builder.append(line)
                            break
                        }
                    }
                }
                builder.toString()
            } catch (e: IOException) {
                Log.w(TAG, "Couldn't read /proc/cpuinfo", e)
                ""
            }
        }

        @JvmStatic
        private external fun modelInfo(path: String, skip: Array<String>): Map<String, Any?>

        @JvmStatic
        private external fun getBackendDevicesInfo(): String

        @JvmStatic
        private external fun setupLog(callback: NativeLogCallback)

        @JvmStatic
        private external fun unsetLog()
    }

    data class InitOptions(
        val modelPath: String,
        val chatTemplate: String? = null,
        val embedding: Boolean = false,
        val embdNormalize: Int = -1,
        val nCtx: Int = 512,
        val nBatch: Int = 512,
        val nUbatch: Int = 512,
        val nThreads: Int = 0,
        val nParallel: Int = 0,
        val nGpuLayers: Int = 0,
        val flashAttn: Boolean = false,
        val flashAttnType: String = "",
        val cacheTypeK: String = "",
        val cacheTypeV: String = "",
        val useMlock: Boolean = true,
        val useMmap: Boolean = true,
        val vocabOnly: Boolean = false,
        val loraPath: String = "",
        val loraScaled: Float = 1.0f,
        val loraAdapters: List<LoraAdapter> = emptyList(),
        val ropeFreqBase: Float = 0f,
        val ropeFreqScale: Float = 0f,
        val poolingType: Int = -1,
        val ctxShift: Boolean = true,
        val kvUnified: Boolean = false,
        val swaFull: Boolean = false,
        val nCpuMoe: Int = 0,
        val useProgressCallback: Boolean = false,
        val noGpuDevices: Boolean = false
    )

    data class LoraAdapter(val path: String, val scaled: Float = 1.0f)

    data class ChatTemplateOptions(
        val chatTemplate: String = "",
        val jsonSchema: String = "",
        val tools: String = "",
        val parallelToolCalls: Boolean = false,
        val toolChoice: String = "",
        val enableThinking: Boolean = false,
        val addGenerationPrompt: Boolean = true,
        val now: String = "",
        val chatTemplateKwargs: String = ""
    )

    interface ProgressListener {
        fun onProgress(percent: Int)
    }

    val contextPtr: Long get() = nativeContext
    var isGpuEnabled: Boolean = false
    var reasonNoGpu: String = ""
    var gpuDeviceName: String = ""
    val modelDetails: Map<String, Any?>

    private var nativeContext: Long
    private val progressCallbackRef: LoadProgressCallback?
    private var isReleased = false

    init {
        if (Companion.isArchNotSupported()) {
            throw IllegalStateException("Only 64-bit architectures are supported")
        }

        val resolvedModelPath = resolveModelPath(options.modelPath, androidContext)
        require(resolvedModelPath.isNotEmpty()) { "Model path cannot be empty" }
        require(isGGUF(resolvedModelPath, androidContext)) { "File is not in GGUF format" }

        progressCallbackRef = if (options.useProgressCallback && progressListener != null) {
            LoadProgressCallback(progressListener)
        } else {
            null
        }

        val loraList = if (options.loraAdapters.isEmpty()) {
            null
        } else {
            ArrayList<Map<String, Any>>(options.loraAdapters.size).apply {
                for (adapter in options.loraAdapters) {
                    add(hashMapOf("path" to adapter.path, "scaled" to adapter.scaled))
                }
            }
        }

        val initParams = HashMap<String, Any?>(32).apply {
            put("model", resolvedModelPath)
            options.chatTemplate?.takeIf { it.isNotEmpty() }?.let { put("chat_template", it) }
            if (options.embedding) put("embedding", true)
            if (options.embdNormalize != -1) put("embd_normalize", options.embdNormalize)
            put("n_ctx", options.nCtx)
            put("n_batch", options.nBatch)
            put("n_ubatch", options.nUbatch)
            if (options.nThreads > 0) put("n_threads", options.nThreads)
            if (options.nParallel > 0) put("n_parallel", options.nParallel)
            if (options.nGpuLayers >= 0) put("n_gpu_layers", options.nGpuLayers)
            if (options.flashAttn) put("flash_attn", true)
            if (options.flashAttnType.isNotEmpty()) put("flash_attn_type", options.flashAttnType)
            if (options.cacheTypeK.isNotEmpty()) put("cache_type_k", options.cacheTypeK)
            if (options.cacheTypeV.isNotEmpty()) put("cache_type_v", options.cacheTypeV)
            put("use_mlock", options.useMlock)
            put("use_mmap", options.useMmap)
            if (options.vocabOnly) put("vocab_only", true)
            if (options.loraPath.isNotEmpty()) put("lora", options.loraPath)
            put("lora_scaled", options.loraScaled)
            loraList?.let { if (it.isNotEmpty()) put("lora_list", it) }
            if (options.ropeFreqBase != 0f) put("rope_freq_base", options.ropeFreqBase)
            if (options.ropeFreqScale != 0f) put("rope_freq_scale", options.ropeFreqScale)
            if (options.poolingType != -1) put("pooling_type", options.poolingType)
            put("ctx_shift", options.ctxShift)
            put("kv_unified", options.kvUnified)
            put("swa_full", options.swaFull)
            if (options.nCpuMoe > 0) put("n_cpu_moe", options.nCpuMoe)
            if (options.noGpuDevices) put("no_gpu_devices", true)
        }

        val initResult = initContext(
            initParams,
            progressCallbackRef
        ) ?: throw IllegalStateException("Failed to initialize context")

        val contextPtrString = (initResult["context"] as? String)?.takeIf { it.isNotEmpty() }
            ?: throw IllegalStateException("Missing native context pointer")

        nativeContext = contextPtrString.toLongOrNull()
            ?: throw IllegalStateException("Invalid native context pointer")
        require(nativeContext != 0L) { "Failed to initialize context" }

        isGpuEnabled = initResult["gpu"] as? Boolean ?: false
        reasonNoGpu = (initResult["reasonNoGPU"] as? String).orEmpty()
        if (!isGpuEnabled && options.noGpuDevices) {
            reasonNoGpu = "GPU devices disabled by user"
        }
        gpuDeviceName = (initResult["gpuDevice"] as? String).orEmpty()

        modelDetails = loadModelDetails(nativeContext)
        Companion.loadedLibrary = (initResult["androidLib"] as? String).orEmpty()
    }

    fun interruptLoad() {
        guardReleased()
        interruptLoad(nativeContext)
    }

    fun getFormattedChat(messages: String, chatTemplate: String = ""): String {
        guardReleased()
        return getFormattedChat(nativeContext, messages, chatTemplate)
    }

    fun getFormattedChatWithJinja(messages: String, options: ChatTemplateOptions = ChatTemplateOptions()): Map<String, Any?> {
        guardReleased()
        val params = HashMap<String, Any?>(8).apply {
            if (options.jsonSchema.isNotEmpty()) put("json_schema", options.jsonSchema)
            if (options.tools.isNotEmpty()) put("tools", options.tools)
            if (options.toolChoice.isNotEmpty()) put("tool_choice", options.toolChoice)
            if (options.now.isNotEmpty()) put("now_str", options.now)
            if (options.chatTemplateKwargs.isNotEmpty()) put("chat_template_kwargs", options.chatTemplateKwargs)
            if (options.parallelToolCalls) put("parallel_tool_calls", true)
            if (options.enableThinking) put("enable_thinking", true)
            if (!options.addGenerationPrompt) put("add_generation_prompt", false)
        }
        return getFormattedChatWithJinja(
            nativeContext,
            messages,
            options.chatTemplate,
            params
        )
    }

    fun completion(
        params: Map<String, Any?>,
        onToken: ((Map<String, Any?>) -> Unit)? = null
    ): Map<String, Any?> {
        guardReleased()
        require(params.containsKey("prompt")) { "Missing required parameter: prompt" }

        val partialCallback = PartialCompletionCallback(onToken)
        val result = doCompletion(nativeContext, params, partialCallback)
        if (result.containsKey("error")) {
            throw IllegalStateException(result["error"] as String)
        }
        return result
    }

    fun stopCompletion() {
        guardReleased()
        stopCompletion(nativeContext)
    }

    fun isPredicting(): Boolean {
        guardReleased()
        return isPredicting(nativeContext)
    }

    fun tokenize(text: String, mediaPaths: List<String> = emptyList()): Map<String, Any?> {
        guardReleased()
        return tokenize(nativeContext, text, mediaPaths.toTypedArray())
    }

    fun detokenize(tokens: List<Int>): String {
        guardReleased()
        return detokenize(nativeContext, tokens.toIntArray())
    }

    fun getEmbedding(text: String, params: Map<String, Any?> = emptyMap()): Map<String, Any?> {
        guardReleased()
        val result = embedding(
            nativeContext,
            text,
            params["embd_normalize"] as? Int ?: -1
        )
        if (result.containsKey("error")) {
            throw IllegalStateException(result["error"] as String)
        }
        return result
    }

    fun embedding(text: String, params: Map<String, Any?> = emptyMap()): Map<String, Any?> =
        getEmbedding(text, params)

    fun getRerank(query: String, documents: List<String>, params: Map<String, Any?> = emptyMap()): Map<String, Any?> {
        guardReleased()
        return rerank(
            nativeContext,
            query,
            documents.toTypedArray(),
            params["normalize"] as? Int ?: -1
        )
    }

    fun rerank(query: String, documents: List<String>, params: Map<String, Any?> = emptyMap()): Map<String, Any?> =
        getRerank(query, documents, params)

    fun bench(pp: Int, tg: Int, pl: Int, nr: Int): String {
        guardReleased()
        return bench(nativeContext, pp, tg, pl, nr)
    }

    fun applyLoraAdapters(loraAdapters: List<Map<String, Any?>>): Int {
        guardReleased()
        val result = applyLoraAdapters(nativeContext, loraAdapters)
        if (result != 0) {
            throw IllegalStateException("Failed to apply lora adapters")
        }
        return result
    }

    fun removeLoraAdapters() {
        guardReleased()
        removeLoraAdapters(nativeContext)
    }

    fun getLoadedLoraAdapters(): List<*> {
        guardReleased()
        return getLoadedLoraAdapters(nativeContext)
    }

    fun loadSession(path: String): Map<String, Any?> {
        guardReleased()
        require(path.isNotEmpty()) { "File path is empty" }
        val file = File(path)
        require(file.exists()) { "File does not exist: $path" }
        val result = loadSession(nativeContext, path)
        if (result.containsKey("error")) {
            throw IllegalStateException(result["error"] as String)
        }
        return result
    }

    fun saveSession(path: String, size: Int): Map<String, Any?> {
        guardReleased()
        require(path.isNotEmpty()) { "File path is empty" }
        return saveSession(nativeContext, path, size)
    }

    fun initMultimodal(params: Map<String, Any?>): Boolean {
        guardReleased()
        val mmprojPath = params["path"] as? String
            ?: throw IllegalArgumentException("mmproj_path is empty")
        val useGpu = params["use_gpu"] as? Boolean ?: true

        if (!mmprojPath.startsWith("content://")) {
            val file = File(mmprojPath)
            require(file.exists()) { "mmproj file does not exist: $mmprojPath" }
        }

        val finalPath = resolveModelPath(mmprojPath, androidContext)
        return initMultimodal(nativeContext, finalPath, useGpu)
    }

    fun isMultimodalEnabled(): Boolean {
        guardReleased()
        return isMultimodalEnabled(nativeContext)
    }

    fun getMultimodalSupport(): Map<String, Any?> {
        guardReleased()
        if (!isMultimodalEnabled()) {
            throw IllegalStateException("Multimodal is not enabled")
        }
        return getMultimodalSupport(nativeContext)
    }

    fun releaseMultimodal() {
        guardReleased()
        releaseMultimodal(nativeContext)
    }

    fun initVocoder(params: Map<String, Any?>): Boolean {
        guardReleased()
        return initVocoder(
            nativeContext,
            params["path"] as? String ?: "",
            params["n_batch"] as? Int ?: 512
        )
    }

    fun isVocoderEnabled(): Boolean {
        guardReleased()
        return isVocoderEnabled(nativeContext)
    }

    fun getFormattedAudioCompletion(speakerJson: String, textToSpeak: String): Map<String, Any?> {
        guardReleased()
        return getFormattedAudioCompletion(nativeContext, speakerJson, textToSpeak)
    }

    fun getAudioCompletionGuideTokens(textToSpeak: String): List<*> {
        guardReleased()
        return getAudioCompletionGuideTokens(nativeContext, textToSpeak)
    }

    fun decodeAudioTokens(tokens: List<Int>): List<*> {
        guardReleased()
        return decodeAudioTokens(nativeContext, tokens.toIntArray())
    }

    fun releaseVocoder() {
        guardReleased()
        releaseVocoder(nativeContext)
    }

    fun queueCompletion(
        params: Map<String, Any?>,
        onToken: ((Int, Map<String, Any?>) -> Unit)? = null,
        onComplete: ((Int, Map<String, Any?>) -> Unit)? = null
    ): Int {
        guardReleased()
        require(params.containsKey("prompt")) { "Missing required parameter: prompt" }

        val partialCallback = PartialCompletionCallback { tokenResult ->
            val requestId = (tokenResult["requestId"] as? Number)?.toInt() ?: -1
            onToken?.invoke(requestId, tokenResult)
        }
        val completionCallback = CompletionCallback { result ->
            val requestId = (result["requestId"] as? Number)?.toInt() ?: -1
            onComplete?.invoke(requestId, result)
        }

        val response = doQueueCompletion(
            nativeContext,
            params,
            partialCallback,
            completionCallback
        )
        if (response.containsKey("error")) {
            throw IllegalStateException(response["error"] as String)
        }

        return response["requestId"] as? Int
            ?: throw IllegalStateException("Failed to queue completion (no requestId)")
    }

    fun queueEmbedding(
        text: String,
        params: Map<String, Any?>,
        onResult: (requestId: Int, embedding: List<Double>) -> Unit
    ): Int {
        guardReleased()
        val callback = EmbeddingCallback { requestId, embedding ->
            onResult(requestId, embedding)
        }
        val response = doQueueEmbedding(
            nativeContext,
            text,
            params["embd_normalize"] as? Int ?: -1,
            callback
        )
        if (response.containsKey("error")) {
            throw IllegalStateException(response["error"] as String)
        }
        return response["requestId"] as? Int
            ?: throw IllegalStateException("Failed to queue embedding (no requestId)")
    }

    fun queueRerank(
        query: String,
        documents: List<String>,
        params: Map<String, Any?>,
        onResults: (requestId: Int, results: List<Map<String, Any?>>) -> Unit
    ): Int {
        guardReleased()
        val callback = RerankCallback { requestId, results ->
            onResults(requestId, results)
        }
        val response = doQueueRerank(
            nativeContext,
            query,
            documents.toTypedArray(),
            params["normalize"] as? Int ?: -1,
            callback
        )
        if (response.containsKey("error")) {
            throw IllegalStateException(response["error"] as String)
        }
        return response["requestId"] as? Int
            ?: throw IllegalStateException("Failed to queue rerank (no requestId)")
    }

    fun cancelRequest(requestId: Int) {
        guardReleased()
        doCancelRequest(nativeContext, requestId)
    }

    fun enableParallelMode(nParallel: Int, nBatch: Int): Boolean {
        guardReleased()
        // Ensure any previous loop is stopped before reconfiguring
        stopProcessingLoop(nativeContext)
        val enabled = enableParallelMode(nativeContext, nParallel, nBatch)
        if (enabled) {
            startProcessingLoop(nativeContext)
        }
        return enabled
    }

    fun disableParallelMode() {
        guardReleased()
        stopProcessingLoop(nativeContext)
        disableParallelMode(nativeContext)
    }

    fun release() {
        if (!isReleased) {
            stopProcessingLoop(nativeContext)
            freeContext(nativeContext)
            isReleased = true
        }
    }

    private fun guardReleased() {
        check(!isReleased) { "LlamaContext has been released" }
    }

    private fun resolveModelPath(path: String, context: Context?): String {
        if (!path.startsWith("content://")) {
            return path
        }
        val uri = Uri.parse(path)
        val pfd: ParcelFileDescriptor? = try {
            context?.contentResolver?.openFileDescriptor(uri, "r")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open content URI $uri", e)
            null
        }
        return pfd?.fd?.toString() ?: path
    }

    private fun isGGUF(filepath: String, context: Context?): Boolean {
        val fileHeader = ByteArray(4)
        var inputStream: InputStream? = null
        return try {
            inputStream = if (filepath.startsWith("content")) {
                val uri = Uri.parse(filepath)
                try {
                    context?.contentResolver?.takePersistableUriPermission(
                        uri,
                        Intent.FLAG_GRANT_READ_URI_PERMISSION
                    )
                } catch (e: SecurityException) {
                    Log.w(TAG, "Persistable permission not granted for URI: $uri")
                }
                context?.contentResolver?.openInputStream(uri)
            } else {
                FileInputStream(filepath)
            }

            val bytesRead = inputStream?.read(fileHeader) ?: 0
            if (bytesRead < 4) {
                false
            } else {
                (0..3).all { fileHeader[it] == ggufHeader[it] }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to check GGUF: ${e.message}")
            false
        } finally {
            try {
                inputStream?.close()
            } catch (_: IOException) {
            }
        }
    }

    @Keep
    private class LoadProgressCallback(
        private val listener: ProgressListener
    ) {
        fun onLoadProgress(progress: Int) {
            listener.onProgress(progress)
        }
    }

    @Keep
    private class PartialCompletionCallback(
        private val onToken: ((Map<String, Any?>) -> Unit)?
    ) {
        fun onPartialCompletion(tokenResult: Map<String, Any?>) {
            onToken?.invoke(tokenResult)
        }
    }

    @Keep
    private class CompletionCallback(
        private val onComplete: (Map<String, Any?>) -> Unit
    ) {
        fun onComplete(result: Map<String, Any?>) {
            onComplete(result)
        }
    }

    @Keep
    private class EmbeddingCallback(
        private val onResult: (Int, List<Double>) -> Unit
    ) {
        fun onResult(requestId: Int, embedding: List<Double>) {
            onResult(requestId, embedding)
        }
    }

    @Keep
    private class RerankCallback(
        private val onResults: (Int, List<Map<String, Any?>>) -> Unit
    ) {
        fun onResults(requestId: Int, results: List<Map<String, Any?>>) {
            onResults(requestId, results)
        }
    }

    @Keep
    private class NativeLogCallback(
        private val logger: ((String, String) -> Unit)?
    ) {
        fun emitNativeLog(level: String, text: String) {
            logger?.invoke(level, text)
        }
    }

    private external fun initContext(
        params: Map<String, Any?>,
        loadProgressCallback: LoadProgressCallback?
    ): Map<String, Any?>?

    private external fun interruptLoad(contextPtr: Long)
    private external fun loadModelDetails(contextPtr: Long): Map<String, Any?>
    private external fun getFormattedChatWithJinja(
        contextPtr: Long,
        messages: String,
        chatTemplate: String,
        params: Map<String, Any?>
    ): Map<String, Any?>
    private external fun getFormattedChat(
        contextPtr: Long,
        messages: String,
        chatTemplate: String
    ): String
    private external fun loadSession(contextPtr: Long, path: String): Map<String, Any?>
    private external fun saveSession(contextPtr: Long, path: String, size: Int): Map<String, Any?>
    private external fun doCompletion(
        contextPtr: Long,
        params: Map<String, Any?>,
        partialCompletionCallback: PartialCompletionCallback
    ): Map<String, Any?>

    private external fun stopCompletion(contextPtr: Long)
    private external fun isPredicting(contextPtr: Long): Boolean
    private external fun tokenize(contextPtr: Long, text: String, mediaPaths: Array<String>): Map<String, Any?>
    private external fun detokenize(contextPtr: Long, tokens: IntArray): String
    private external fun embedding(contextPtr: Long, text: String, embdNormalize: Int): Map<String, Any?>
    private external fun rerank(contextPtr: Long, query: String, documents: Array<String>, normalize: Int): Map<String, Any?>
    private external fun bench(contextPtr: Long, pp: Int, tg: Int, pl: Int, nr: Int): String
    private external fun applyLoraAdapters(contextPtr: Long, loraAdapters: List<*>): Int
    private external fun removeLoraAdapters(contextPtr: Long)
    private external fun getLoadedLoraAdapters(contextPtr: Long): List<*>
    private external fun freeContext(contextPtr: Long)
    private external fun initMultimodal(contextPtr: Long, mmprojPath: String, useGpu: Boolean): Boolean
    private external fun isMultimodalEnabled(contextPtr: Long): Boolean
    private external fun getMultimodalSupport(contextPtr: Long): Map<String, Any?>
    private external fun releaseMultimodal(contextPtr: Long)
    private external fun initVocoder(contextPtr: Long, vocoderModelPath: String, batchSize: Int): Boolean
    private external fun isVocoderEnabled(contextPtr: Long): Boolean
    private external fun getFormattedAudioCompletion(contextPtr: Long, speakerJsonStr: String, textToSpeak: String): Map<String, Any?>
    private external fun getAudioCompletionGuideTokens(contextPtr: Long, textToSpeak: String): List<*>
    private external fun decodeAudioTokens(contextPtr: Long, tokens: IntArray): List<*>
    private external fun releaseVocoder(contextPtr: Long)
    private external fun doQueueCompletion(
        contextPtr: Long,
        params: Map<String, Any?>,
        partialCompletionCallback: PartialCompletionCallback,
        completionCallback: CompletionCallback
    ): Map<String, Any?>
    private external fun doCancelRequest(contextPtr: Long, requestId: Int)
    private external fun doQueueEmbedding(
        contextPtr: Long,
        text: String,
        embdNormalize: Int,
        callback: EmbeddingCallback
    ): Map<String, Any?>
    private external fun doQueueRerank(
        contextPtr: Long,
        query: String,
        documents: Array<String>,
        normalize: Int,
        callback: RerankCallback
    ): Map<String, Any?>
    private external fun enableParallelMode(contextPtr: Long, nParallel: Int, nBatch: Int): Boolean
    private external fun disableParallelMode(contextPtr: Long)
    private external fun startProcessingLoop(contextPtr: Long)
    private external fun stopProcessingLoop(contextPtr: Long)
}
