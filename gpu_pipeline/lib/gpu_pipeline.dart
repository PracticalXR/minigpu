/// GPU-accelerated streaming pipelines.
///
/// A [StageBuilder] chains built-in tensor ops, WGSL compute shaders and Dart
/// functions into a [DynamicStage]. A [Pipeline] runs its stages on every
/// frame a [MediaStream] emits. [PipelineLoader] builds pipelines from a JSON
/// network config, and an [AssetManager] caches the assets stages declare.
library;

export 'src/pipeline.dart';
export 'src/pipeline_event.dart';
export 'src/pipeline_stage.dart';
export 'src/pipeline_stage_builder.dart';
export 'src/operation/operation.dart';
export 'src/pipeline_dynamic_stage.dart';
export 'src/pipeline_properties.dart';
export 'src/stages/stream_merge_stage.dart';
export 'src/media_stream.dart';
export 'src/thread_monitor.dart';
export 'src/pipeline_assets.dart'
    show AssetEvent, AssetManager, AssetType, DisposableToken, StaticAsset;
export 'src/serialization/patch_executor.dart'
    show RuntimePatchApplyResult, RuntimePatchExecutor;
export 'src/serialization/pipeline_loader.dart'
    show PipelineLoadResult, PipelineLoader, PipelineValidationException;
export 'src/serialization/pipeline_models.dart'
    show
        AssetConfig,
        ConnectionConfig,
        InterPipelineLinkConfig,
        InterPipelineLinkMode,
        NetworkConfig,
        OperationConfig,
        PipelineConfig,
        PortConfig,
        ShapeExprConfig,
        StageConfig;
export 'src/serialization/pipeline_patch.dart'
    show
        AddOperationOp,
        PatchApplier,
        PatchEnvelope,
        PatchOp,
        ReplaceStageOp,
        SetParamOp;
export 'src/serialization/pipeline_serializer.dart'
    show PipelineSerializationResult, PipelineSerializer;
export 'src/serialization/pipeline_validator.dart'
    show PipelineValidator, ValidationIssue, ValidationReport;
export 'src/serialization/runtime_builder.dart'
    show RuntimeBuildResult, RuntimeBuilder;
