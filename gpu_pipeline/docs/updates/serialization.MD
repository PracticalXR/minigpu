# Serialization design (v1)

This describes the JSON schema and runtime/validation behavior implemented in the current codebase.

## Top-level types (JSON keys are snake_case)

- NetworkConfig
  - schema_version: number (required; current = 1)
  - pipelines: PipelineConfig[] (required)
  - inter_pipeline_links?: InterPipelineLinkConfig[]
  - assets?: AssetConfig[]
  - notes?: string

- PipelineConfig
  - type: "pipeline"
  - id: string (required)
  - queue: object (required)
    - strategy: "latest_only" | "latest_n" | "drop_on_overrun" | "batched" | "process_all"
    - max_items_per_stream?: number
    - max_total_items?: number
    - batch_size?: number
  - enable_benchmarking?: bool
  - debug_enabled?: bool
  - stages: StageConfig[] (required)
  - connections: ConnectionConfig[] (required; can be empty)
  - parameters?: { [k: string]: any }
  - metadata?: { [k: string]: any }

- StageConfig
  - type: "stage"
  - stage_id: string (required; unique within pipeline)
  - stage_kind: string (required; currently "DynamicStage")
  - order?: number
  - execution_target?: string
  - resource?: object
  - inputs?: PortConfig[]
  - outputs?: PortConfig[]
  - parameters?: { [k: string]: any }
  - operations?: OperationConfig[] (executed in order)
  - assets?: string[]

- OperationConfig
  - type: "operation"
  - op_kind: "tensor_op" | "dynamic_cpu" | "shader"
  - input_keys?: string[]
  - output_keys?: string[]
  - arguments?: { args?: any[]; [name: string]: any }  // positional in args + named entries
  - named_arguments?: { [name: string]: any }
  - operation?: string                // tensor_op id (e.g., "addScalar")
  - dart_code?: string                // for dynamic_cpu
  - wgsl?: string                     // for shader
  - shader_parameters?: { [k: string]: any } // for shader
  - workgroup_size?: [number, number, number] // for shader

- ConnectionConfig
  - from_stage: string
  - to_stage: string
  - from_output?: string
  - to_input?: string

- PortConfig
  - name: string
  - port_type: string
  - shape_policy?: string
  - size_policy?: string
  - shape_expr?: ShapeExprConfig
  - min_dimensions?: number
  - max_dimensions?: number
  - required?: bool

- ShapeExprConfig
  - op: "fixed" | "same_as" | "concat_along" | "broadcast_pair" | "reshape_flat"
  - ...other op-specific fields (stored verbatim in raw for evaluation)

- AssetConfig
  - type: "asset"
  - name: string
  - asset_type: string
  - source: object
  - loading?: object
  - metadata?: object

- InterPipelineLinkConfig
  - from: { pipeline: string, output: string }
  - to: { pipeline: string, input: string }
  - mode: string

## Argument references

At runtime, strings in arguments/named_arguments are resolved when prefixed:

- param:<name> → stage.parameters[name]
- tensor:<inputKey> → input tensor object
- shape:<inputKey> → List<int> shape of input tensor
- size:<inputKey> → total element count of input tensor

The tests currently use only param:.

## Validation

Emitted error codes (covered by tests and code):

- UNSUPPORTED_SCHEMA_VERSION
- DUP_STAGE_ID
- OP_KIND_MISSING
- OP_NAME_MISSING (tensor_op missing operation)
- CONNECTION_UNKNOWN_STAGE
- SHAPE_* (shape_expr evaluation errors)

Note: shape_expr validation currently attempts evaluation without real input shapes. Prefer fixed/param-only expressions at authoring time; more context-aware validation may be added.

## Runtime: loading and execution

- Roundtrip:
  - NetworkConfig.toJson()/fromJson(Map) and encode()/decode(String) use snake_case keys.
- Load + build:
  - PipelineLoader().loadFromJson(jsonString) → { config, validation, runtime }
  - The builder honors queue.strategy and optional limits; connections are attached to runtime pipelines.
- Execute:
  - await pipeline.runOnce({ "input": Tensor }) → Map<String, Tensor>
- Live parameter patch:
  - Patch envelope JSON uses snake_case (op: "set_param", stage_id/name/value).
  - RuntimePatchExecutor applies set_param to live stages.

## Live patch operations (JSON)

Patches are envelopes applied to a single pipeline.

- Envelope
  - type: "patch"
  - pipeline_id: string
  - ops: PatchOp[]

Supported ops:

- set_param
  - stage_id: string
  - name: string
  - value: any
  - Effect: updates stage.parameters[name] at runtime; no rebuild.

- replace_stage
  - target_stage_id: string
  - stage: StageConfig
  - Effect: replaces the target stage with the provided StageConfig. If the new
    stage.stage_id differs, connections referencing the old id are rewritten.
    Triggers a rebuild and hot-swap of the pipeline.

- add_operation
  - stage_id: string
  - operation: OperationConfig
  - index?: number (default append; <0 coerces to 0; >len coerces to append)
  - Effect: inserts an operation into the stage’s operations list at the given
    index (or appends). Triggers a rebuild and hot-swap.

Notes:
- Structural ops (replace_stage, add_operation, future remove/insert/reorder stage)
  rebuild the pipeline internally and replace the live instance.
- Param-only patches are applied in-place without rebuild.

## JSON examples

- Dynamic CPU add 5:

```json
{
  "schema_version": 1,
  "pipelines": [
    {
      "id": "p",
      "queue": { "strategy": "latest_only" },
      "stages": [
        {
          "stage_id": "dyn",
          "stage_kind": "DynamicStage",
          "operations": [
            {
              "op_kind": "dynamic_cpu",
              "input_keys": ["input"],
              "output_keys": ["output"],
              "dart_code": "Map<String, TypedData> main(Map<String, TypedData> inputs, Map<String, List<int>> ranks, Map<String, dynamic> parameters) { final input = inputs['input'] as Float32List; final out = Float32List(input.length); for (var i=0;i<input.length;i++){ out[i] = input[i] + 5; } return {'output': out}; }"
            }
          ]
        }
      ],
      "connections": []
    }
  ]
}
```

- Two-stage tensor ops with param reference and connection:

```json
{
  "schema_version": 1,
  "pipelines": [
    {
      "id": "p",
      "queue": { "strategy": "latest_only" },
      "stages": [
        {
          "stage_id": "s1",
          "stage_kind": "DynamicStage",
          "parameters": { "delta": 2.0 },
          "operations": [
            {
              "op_kind": "tensor_op",
              "operation": "addScalar",
              "input_keys": ["input"],
              "output_keys": ["s1_out"],
              "arguments": { "args": ["param:delta"] }
            }
          ]
        },
        {
          "stage_id": "s2",
          "stage_kind": "DynamicStage",
          "operations": [
            {
              "op_kind": "tensor_op",
              "operation": "addScalar",
              "input_keys": ["mid"],
              "output_keys": ["output"],
              "arguments": { "args": [2.0] }
            }
          ]
        }
      ],
      "connections": [
        { "from_stage": "s1", "to_stage": "s2", "from_output": "s1_out", "to_input": "mid" }
      ]
    }
  ]
}
```

## Notes and planned extensions

- Future patches (replace_stage, add_operation, etc.) are scaffolded in design but not implemented yet.
- Inter-pipeline links are parsed but not executed (warning emitted during build).
