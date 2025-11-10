package com.afreecatv.mobile.sdk.aiEdgeLayer.translator

import android.content.Context
import android.util.Log
import com.afreecatv.mobile.sdk.aiEdgeLayer.translator.LlamaContext.ChatTemplateOptions
import com.afreecatv.mobile.sdk.aiEdgeLayer.translator.LlamaContext.InitOptions
import com.afreecatv.mobile.sdk.aiEdgeLayer.translator.LlamaContext.LoraAdapter
import com.afreecatv.mobile.sdk.aiEdgeLayer.translator.LlamaContext.ProgressListener
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.util.Random
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.absoluteValue

/**
 * Android wrapper for LlamaContext that provides high-level APIs with Flow support.
 */
class LlamaAndroid(private val applicationContext: Context? = null) {

    companion object {
        private const val NAME = "LlamaAndroid"
    }

    private val contexts = ConcurrentHashMap<Int, LlamaContext>()
    private val eventFlows = ConcurrentHashMap<Int, MutableSharedFlow<Pair<String, Any?>>>()
    private var llamaContextLimit = 1

    fun setContextLimit(limit: Int) {
        llamaContextLimit = limit
    }

    fun initContext(params: Map<String, Any?>): Map<String, Any?> {
        val id = Random().nextInt().absoluteValue
        return try {
            if (contexts.size >= llamaContextLimit) {
                throw Exception("Context limit reached")
            }

            val eventFlow = eventFlows.getOrPut(id) { MutableSharedFlow(extraBufferCapacity = 64) }
            val options = params.toInitOptions()
            val progressListener = if (options.useProgressCallback) {
                object : ProgressListener {
                    override fun onProgress(percent: Int) {
                        eventFlow.tryEmit("progress" to percent)
                    }
                }
            } else null

            val llamaContext = LlamaContext(id, applicationContext, options, progressListener)
            contexts[id] = llamaContext

            mapOf(
                "contextId" to id,
                "gpu" to llamaContext.isGpuEnabled,
                "reasonNoGPU" to llamaContext.reasonNoGpu,
                "gpuDevice" to llamaContext.gpuDeviceName,
                "model" to llamaContext.modelDetails,
                "loadedLibrary" to LlamaContext.loadedNativeLibrary(),
                "contextPtr" to llamaContext.contextPtr
            )
        } catch (e: Exception) {
            Log.e(NAME, "Error initializing context", e)
            eventFlows.remove(id)
            mapOf("error" to (e.message ?: "Unknown error"))
        }
    }

    fun releaseContext(id: Int) {
        contexts.remove(id)?.release()
        eventFlows.remove(id)
    }

    fun releaseAllContexts() {
        contexts.values.forEach { it.release() }
        contexts.clear()
        eventFlows.clear()
    }

    fun getFormattedChatWithJinja(
        id: Int,
        messages: String,
        chatTemplate: String,
        params: Map<String, Any?> = emptyMap(),
        addGenerationPrompt: Boolean = true,
        nowStr: String = "",
        chatTemplateKwargs: String = ""
    ): Flow<Map<String, Any?>> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            val options = ChatTemplateOptions(
                chatTemplate = chatTemplate,
                jsonSchema = params.getString("json_schema"),
                tools = params.getString("tools"),
                parallelToolCalls = params.getBool("parallel_tool_calls", false),
                toolChoice = params.getString("tool_choice"),
                enableThinking = params.getBool("enable_thinking", false),
                addGenerationPrompt = addGenerationPrompt,
                now = nowStr.ifEmpty { params.getString("now") },
                chatTemplateKwargs = chatTemplateKwargs.ifEmpty { params.getString("chat_template_kwargs") }
            )
            emit(context.getFormattedChatWithJinja(messages, options))
        } catch (e: Exception) {
            Log.e(NAME, "Error formatting chat with Jinja", e)
            emit(mapOf("error" to (e.message ?: "Unknown error")))
        }
    }.flowOn(Dispatchers.IO)

    fun getFormattedChat(id: Int, messages: String, chatTemplate: String): Flow<String> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.getFormattedChat(messages, chatTemplate))
        } catch (e: Exception) {
            Log.e(NAME, "Error formatting chat", e)
            emit("")
        }
    }.flowOn(Dispatchers.IO)

    fun loadSession(id: Int, path: String): Flow<Map<String, Any?>> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.loadSession(path))
        } catch (e: Exception) {
            Log.e(NAME, "Error loading session", e)
            emit(mapOf("error" to (e.message ?: "Unknown error")))
        }
    }.flowOn(Dispatchers.IO)

    fun saveSession(id: Int, path: String, size: Int): Flow<Map<String, Any?>> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.saveSession(path, size))
        } catch (e: Exception) {
            Log.e(NAME, "Error saving session", e)
            emit(mapOf("error" to (e.message ?: "Unknown error")))
        }
    }.flowOn(Dispatchers.IO)

    fun setEventCollector(id: Int, scope: CoroutineScope): MutableSharedFlow<Pair<String, Any?>>? {
        return eventFlows[id]
    }

    fun unsetEventCollector(id: Int) {
        eventFlows.remove(id)
    }

    fun launchCompletion(id: Int, params: Map<String, Any?>): Flow<Map<String, Any?>> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            val flow = eventFlows[id]
            val result = context.completion(params) { token ->
                flow?.tryEmit("token" to token)
            }
            flow?.tryEmit("completion" to result)
            emit(result)
        } catch (e: Exception) {
            Log.e(NAME, "Error during completion", e)
            emit(mapOf("error" to (e.message ?: "Unknown error")))
        }
    }.flowOn(Dispatchers.IO)

    suspend fun stopCompletion(id: Int) = withContext(Dispatchers.IO) {
        val context = contexts[id] ?: throw Exception("Context not found")
        context.stopCompletion()
    }

    fun isPredicting(id: Int): Boolean = contexts[id]?.isPredicting() ?: false

    fun tokenize(id: Int, text: String, mediaPaths: List<String> = emptyList()): Flow<Map<String, Any?>> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.tokenize(text, mediaPaths))
        } catch (e: Exception) {
            Log.e(NAME, "Error tokenizing text", e)
            emit(mapOf("error" to (e.message ?: "Unknown error")))
        }
    }.flowOn(Dispatchers.IO)

    fun detokenize(id: Int, tokens: List<Int>): Flow<String> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.detokenize(tokens))
        } catch (e: Exception) {
            Log.e(NAME, "Error detokenizing tokens", e)
            emit("")
        }
    }.flowOn(Dispatchers.IO)

    fun getEmbedding(id: Int, text: String, params: Map<String, Any?> = emptyMap()): Flow<Map<String, Any?>> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.getEmbedding(text, params))
        } catch (e: Exception) {
            Log.e(NAME, "Error getting embedding", e)
            emit(mapOf("error" to (e.message ?: "Unknown error")))
        }
    }.flowOn(Dispatchers.IO)

    fun getRerank(id: Int, query: String, documents: List<String>, params: Map<String, Any?> = emptyMap()): Flow<Map<String, Any?>> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.getRerank(query, documents, params))
        } catch (e: Exception) {
            Log.e(NAME, "Error getting rerank", e)
            emit(mapOf("error" to (e.message ?: "Unknown error")))
        }
    }.flowOn(Dispatchers.IO)

    fun bench(id: Int, pp: Int, tg: Int, pl: Int, nr: Int): Flow<String> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.bench(pp, tg, pl, nr))
        } catch (e: Exception) {
            Log.e(NAME, "Error running benchmark", e)
            emit("")
        }
    }.flowOn(Dispatchers.IO)

    fun applyLoraAdapters(id: Int, loraAdapters: List<*>): Flow<Int> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            val adapters = loraAdapters.mapNotNull { entry ->
                (entry as? Map<*, *>)?.let { map ->
                    map as Map<String, Any?>
                }
            }
            emit(context.applyLoraAdapters(adapters))
        } catch (e: Exception) {
            Log.e(NAME, "Error applying LoRA adapters", e)
            emit(-1)
        }
    }.flowOn(Dispatchers.IO)

    fun removeLoraAdapters(id: Int): Flow<Unit> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            context.removeLoraAdapters()
            emit(Unit)
        } catch (e: Exception) {
            Log.e(NAME, "Error removing LoRA adapters", e)
        }
    }.flowOn(Dispatchers.IO)

    fun getLoadedLoraAdapters(id: Int): Flow<List<*>> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.getLoadedLoraAdapters())
        } catch (e: Exception) {
            Log.e(NAME, "Error getting loaded LoRA adapters", e)
            emit(emptyList<Any>())
        }
    }.flowOn(Dispatchers.IO)

    fun queueEmbedding(id: Int, text: String, params: Map<String, Any?> = emptyMap()): Flow<Int> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            val eventFlow = eventFlows[id]
            val requestId = context.queueEmbedding(text, params) { reqId, embedding ->
                val payload = mapOf(
                    "requestId" to reqId,
                    "embedding" to embedding
                )
                eventFlow?.tryEmit("embedding" to payload)
            }
            emit(requestId)
        } catch (e: Exception) {
            Log.e(NAME, "Error queuing embedding", e)
            emit(-1)
        }
    }.flowOn(Dispatchers.IO)

    fun queueRerank(
        id: Int,
        query: String,
        documents: List<String>,
        params: Map<String, Any?> = emptyMap()
    ): Flow<Int> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            val eventFlow = eventFlows[id]
            val requestId = context.queueRerank(query, documents, params) { reqId, results ->
                val payload = mapOf(
                    "requestId" to reqId,
                    "results" to results
                )
                eventFlow?.tryEmit("rerank" to payload)
            }
            emit(requestId)
        } catch (e: Exception) {
            Log.e(NAME, "Error queuing rerank", e)
            emit(-1)
        }
    }.flowOn(Dispatchers.IO)

    fun queueCompletion(id: Int, params: Map<String, Any?>): Flow<Int> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            val eventFlow = eventFlows[id]
            val requestId = context.queueCompletion(
                params,
                onToken = { reqId, token ->
                    val payload = HashMap(token)
                    payload["requestId"] = reqId
                    eventFlow?.tryEmit("token" to payload)
                },
                onComplete = { reqId, result ->
                    val payload = HashMap(result)
                    payload["requestId"] = reqId
                    eventFlow?.tryEmit("completion" to payload)
                }
            )
            emit(requestId)
        } catch (e: Exception) {
            Log.e(NAME, "Error queueing completion", e)
            emit(-1)
        }
    }.flowOn(Dispatchers.IO)

    fun initMultimodal(id: Int, params: Map<String, Any?>): Flow<Boolean> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.initMultimodal(params))
        } catch (e: Exception) {
            Log.e(NAME, "Error initializing multimodal", e)
            emit(false)
        }
    }.flowOn(Dispatchers.IO)

    fun isMultimodalEnabled(id: Int): Boolean = contexts[id]?.isMultimodalEnabled() ?: false

    fun getMultimodalSupport(id: Int): Flow<Map<String, Any?>> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.getMultimodalSupport())
        } catch (e: Exception) {
            Log.e(NAME, "Error getting multimodal support", e)
            emit(mapOf("error" to (e.message ?: "Unknown error")))
        }
    }.flowOn(Dispatchers.IO)

    fun releaseMultimodal(id: Int): Flow<Unit> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            context.releaseMultimodal()
            emit(Unit)
        } catch (e: Exception) {
            Log.e(NAME, "Error releasing multimodal", e)
        }
    }.flowOn(Dispatchers.IO)

    fun initVocoder(id: Int, params: Map<String, Any?>): Flow<Boolean> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.initVocoder(params))
        } catch (e: Exception) {
            Log.e(NAME, "Error initializing vocoder", e)
            emit(false)
        }
    }.flowOn(Dispatchers.IO)

    fun isVocoderEnabled(id: Int): Boolean = contexts[id]?.isVocoderEnabled() ?: false

    fun getFormattedAudioCompletion(id: Int, speakerJsonStr: String, textToSpeak: String): Flow<Map<String, Any?>> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.getFormattedAudioCompletion(speakerJsonStr, textToSpeak))
        } catch (e: Exception) {
            Log.e(NAME, "Error getting formatted audio completion", e)
            emit(mapOf("error" to (e.message ?: "Unknown error")))
        }
    }.flowOn(Dispatchers.IO)

    fun getAudioCompletionGuideTokens(id: Int, textToSpeak: String): Flow<List<*>> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.getAudioCompletionGuideTokens(textToSpeak))
        } catch (e: Exception) {
            Log.e(NAME, "Error getting audio completion guide tokens", e)
            emit(emptyList<Any>())
        }
    }.flowOn(Dispatchers.IO)

    fun decodeAudioTokens(id: Int, tokens: List<Int>): Flow<List<*>> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            emit(context.decodeAudioTokens(tokens))
        } catch (e: Exception) {
            Log.e(NAME, "Error decoding audio tokens", e)
            emit(emptyList<Any>())
        }
    }.flowOn(Dispatchers.IO)

    fun releaseVocoder(id: Int): Flow<Unit> = flow {
        try {
            val context = contexts[id] ?: throw Exception("Context not found")
            context.releaseVocoder()
            emit(Unit)
        } catch (e: Exception) {
            Log.e(NAME, "Error releasing vocoder", e)
        }
    }.flowOn(Dispatchers.IO)

    fun interruptLoad(id: Int) {
        contexts[id]?.interruptLoad()
    }

    fun toggleNativeLog(enabled: Boolean) {
        LlamaContext.toggleNativeLog(enabled)
    }

    fun getCpuFeatures(): String = LlamaContext.cpuFeatures()

    fun isArchSupported(): Boolean = LlamaContext.isArchSupported()

    fun enableParallelMode(id: Int, nParallel: Int, nBatch: Int): Boolean {
        val context = contexts[id] ?: return false
        return context.enableParallelMode(nParallel, nBatch)
    }

    fun disableParallelMode(id: Int) {
        contexts[id]?.disableParallelMode()
    }

    private fun Map<String, Any?>.getBool(key: String, default: Boolean): Boolean = when (val value = this[key]) {
        is Boolean -> value
        is Number -> value.toInt() != 0
        is String -> value.equals("true", ignoreCase = true)
        else -> default
    }

    private fun Map<String, Any?>.getInt(key: String, default: Int): Int = when (val value = this[key]) {
        is Number -> value.toInt()
        is String -> value.toIntOrNull() ?: default
        else -> default
    }

    private fun Map<String, Any?>.getFloat(key: String, default: Float): Float = when (val value = this[key]) {
        is Number -> value.toFloat()
        is String -> value.toFloatOrNull() ?: default
        else -> default
    }

    private fun Map<String, Any?>.getString(key: String, default: String = ""): String =
        (this[key] as? String)?.takeIf { it.isNotEmpty() } ?: default

    private fun Map<String, Any?>.toInitOptions(): InitOptions {
        val modelPath = this["model"] as? String
            ?: throw IllegalArgumentException("Missing required parameter: model")

        val loraAdapters = (this["lora_list"] as? List<*>)?.mapNotNull { entry ->
            (entry as? Map<*, *>)?.let {
                val path = it["path"] as? String ?: return@let null
                val scaled = (it["scaled"] as? Number)?.toFloat() ?: 1.0f
                LoraAdapter(path, scaled)
            }
        } ?: emptyList()

        return InitOptions(
            modelPath = modelPath,
            chatTemplate = this["chat_template"] as? String,
            embedding = getBool("embedding", false),
            embdNormalize = getInt("embd_normalize", -1),
            nCtx = getInt("n_ctx", 512),
            nBatch = getInt("n_batch", 512),
            nUbatch = getInt("n_ubatch", 512),
            nThreads = getInt("n_threads", 0),
            nParallel = getInt("n_parallel", 0),
            nGpuLayers = getInt("n_gpu_layers", 0),
            flashAttn = getBool("flash_attn", false),
            flashAttnType = getString("flash_attn_type"),
            cacheTypeK = getString("cache_type_k", "f16"),
            cacheTypeV = getString("cache_type_v", "f16"),
            useMlock = getBool("use_mlock", true),
            useMmap = getBool("use_mmap", true),
            vocabOnly = getBool("vocab_only", false),
            loraPath = getString("lora"),
            loraScaled = (this["lora_scaled"] as? Number)?.toFloat() ?: 1.0f,
            loraAdapters = loraAdapters,
            ropeFreqBase = getFloat("rope_freq_base", 0f),
            ropeFreqScale = getFloat("rope_freq_scale", 0f),
            poolingType = getInt("pooling_type", -1),
            ctxShift = getBool("ctx_shift", true),
            kvUnified = getBool("kv_unified", false),
            swaFull = getBool("swa_full", false),
            nCpuMoe = getInt("n_cpu_moe", 0),
            useProgressCallback = getBool("use_progress_callback", false),
            noGpuDevices = getBool("no_gpu_devices", false)
        )
    }
}
