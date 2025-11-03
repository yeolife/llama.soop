#import <Foundation/Foundation.h>

#ifdef __cplusplus
namespace rnllama {
    struct llama_rn_context;
}
#endif

@interface RNLlamaContext : NSObject {
    bool is_metal_enabled;
    bool is_model_loaded;
    NSString * reason_no_metal;
    NSString * gpu_device_name;

    void (^onProgress)(unsigned int progress);

#ifdef __cplusplus
    rnllama::llama_rn_context * llama;
#else
    void * llama;
#endif
}

+ (void)toggleNativeLog:(BOOL)enabled onEmitLog:(void (^)(NSString *level, NSString *text))onEmitLog;
+ (NSDictionary *)modelInfo:(NSString *)path skip:(NSArray *)skip;
+ (NSString *)getBackendDevicesInfo;
+ (instancetype)initWithParams:(NSDictionary *)params onProgress:(void (^)(unsigned int progress))onProgress;
- (void)interruptLoad;
- (bool)isMetalEnabled;
- (NSString *)reasonNoMetal;
- (NSString *)gpuDeviceName;
- (NSDictionary *)modelInfo;
- (bool)isModelLoaded;
- (bool)isPredicting;
- (bool)initMultimodal:(NSDictionary *)params;
- (NSDictionary *)getMultimodalSupport;
- (bool)isMultimodalEnabled;
- (void)releaseMultimodal;
- (NSDictionary *)completion:(NSDictionary *)params;
- (void)stopCompletion;
- (NSNumber *)queueCompletion:(NSDictionary *)params onToken:(void (^)(NSMutableDictionary *tokenResult))onToken onComplete:(void (^)(NSDictionary *result))onComplete;
- (NSNumber *)queueEmbedding:(NSString *)text params:(NSDictionary *)params onResult:(void (^)(int32_t requestId, NSArray *embedding))onResult;
- (NSNumber *)queueRerank:(NSString *)query documents:(NSArray<NSString *> *)documents params:(NSDictionary *)params onResults:(void (^)(int32_t requestId, NSArray *results))onResults;
- (void)cancelRequest:(NSNumber *)requestId;
- (BOOL)enableParallelMode:(int)nParallel nBatch:(int)nBatch;
- (void)disableParallelMode;
- (NSDictionary *)tokenize:(NSString *)text mediaPaths:(NSArray *)mediaPaths;
- (NSString *)detokenize:(NSArray *)tokens;
- (NSDictionary *)embedding:(NSString *)text params:(NSDictionary *)params;
- (NSArray *)rerank:(NSString *)query documents:(NSArray<NSString *> *)documents params:(NSDictionary *)params;
- (NSDictionary *)getFormattedChatWithJinja:(NSString *)messages
                           withChatTemplate:(NSString *)chatTemplate
                             withJsonSchema:(NSString *)jsonSchema
                                  withTools:(NSString *)tools
                      withParallelToolCalls:(BOOL)parallelToolCalls
                             withToolChoice:(NSString *)toolChoice
                         withEnableThinking:(BOOL)enableThinking
                    withAddGenerationPrompt:(BOOL)addGenerationPrompt
                                    withNow:(NSString *)nowStr
                     withChatTemplateKwargs:(NSString *)chatTemplateKwargs;
- (NSString *)getFormattedChat:(NSString *)messages withChatTemplate:(NSString *)chatTemplate;
- (NSDictionary *)loadSession:(NSString *)path;
- (int)saveSession:(NSString *)path size:(int)size;
- (NSString *)bench:(int)pp tg:(int)tg pl:(int)pl nr:(int)nr;
- (void)applyLoraAdapters:(NSArray *)loraAdapters;
- (void)removeLoraAdapters;
- (NSArray *)getLoadedLoraAdapters;
- (bool)initVocoder:(NSDictionary *)params;
- (bool)isVocoderEnabled;
- (NSDictionary *)getFormattedAudioCompletion:(NSString *)speakerJsonStr textToSpeak:(NSString *)textToSpeak;
- (NSArray *)getAudioCompletionGuideTokens:(NSString *)textToSpeak;
- (NSArray *)decodeAudioTokens:(NSArray *)tokens;
- (void)releaseVocoder;
- (void)invalidate;

@end
