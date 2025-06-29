import { useEffect, useState } from 'react'
import { Image } from 'react-native'
import { OnnxModule } from './OnnxModule'

type TypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | BigInt64Array
  | BigUint64Array

declare global {
  /**
   * Loads the ONNX Model into memory. Path is fetchable resource, e.g.:
   * http://192.168.8.110:8081/assets/assets/model.onnx?platform=ios&hash=32e9958c83e5db7d0d693633a9f0b175
   */
  // eslint-disable-next-line no-var
  var __loadOnnxModel: (
    path: string,
    provider: OnnxModelProvider
  ) => Promise<OnnxModel>
}

// Installs the JSI bindings into the global namespace.
console.log('Installing ONNX bindings...')
const result = OnnxModule.install() as boolean
if (result !== true)
  console.error('Failed to install ONNX Runtime bindings!')

console.log('ONNX Runtime successfully installed!')

export type OnnxModelProvider =
  | 'cpu'
  | 'coreml'
  | 'nnapi'
  | 'gpu'

export interface OnnxTensor {
  /**
   * The name of the Tensor.
   */
  name: string
  /**
   * The data-type all values of this Tensor are represented in.
   */
  dataType:
    | 'bool'
    | 'uint8'
    | 'int8'
    | 'int16'
    | 'int32'
    | 'int64'
    | 'float16'
    | 'float32'
    | 'float64'
    | 'invalid'
  /**
   * The shape of the data from this tensor.
   */
  shape: number[]
}

export interface OnnxModel {
  /**
   * The computation provider used by this Model.
   */
  provider: OnnxModelProvider
  /**
   * Run the ONNX Model with the given input buffer.
   * The input buffer has to match the input tensor's shape.
   */
  run(input: TypedArray[]): Promise<TypedArray[]>
  /**
   * Synchronously run the ONNX Model with the given input buffer.
   * The input buffer has to match the input tensor's shape.
   */
  runSync(input: TypedArray[]): TypedArray[]

  /**
   * All input tensors of this ONNX Model.
   */
  inputs: OnnxTensor[]
  /**
   * All output tensors of this ONNX Model.
   * The user is responsible for correctly interpreting this data.
   */
  outputs: OnnxTensor[]
}

// In React Native, `require(..)` returns a number.
type Require = number // ReturnType<typeof require>
type ModelSource = Require | { url: string }

export type OnnxPlugin =
  | {
      model: OnnxModel
      state: 'loaded'
    }
  | {
      model: undefined
      state: 'loading'
    }
  | {
      model: undefined
      error: Error
      state: 'error'
    }

/**
 * Load an ONNX Model from the given `.onnx` asset.
 *
 * * If you are passing in a `.onnx` model from your app's bundle using `require(..)`, make sure to add `onnx` as an asset extension to `metro.config.js`!
 * * If you are passing in a `{ url: ... }`, make sure the URL points directly to a `.onnx` model. This can either be a web URL (`http://..`/`https://..`), or a local file (`file://..`).
 *
 * @param source The `.onnx` model in form of either a `require(..)` statement or a `{ url: string }`.
 * @param provider The provider to use for computations. Uses the standard CPU provider per default. The `coreml` or `gpu` providers are hardware-accelerated.
 * @returns The loaded Model.
 */
export function loadOnnxModel(
  source: ModelSource,
  provider: OnnxModelProvider = 'cpu'
): Promise<OnnxModel> {
  let uri: string
  if (typeof source === 'number') {
    console.log(`Loading ONNX Model ${source}`)
    const asset = Image.resolveAssetSource(source)
    uri = asset.uri
    console.log(`Resolved ONNX Model path: ${asset.uri}`)
  } else if (typeof source === 'object' && 'url' in source) {
    uri = source.url
  } else {
    throw new Error(
      'ONNX: Invalid source passed! Source should be either a React Native require(..) or a `{ url: string }` object!'
    )
  }
  return global.__loadOnnxModel(uri, provider)
}

/**
 * Load an ONNX Model from the given `.onnx` asset into a React State.
 *
 * * If you are passing in a `.onnx` model from your app's bundle using `require(..)`, make sure to add `onnx` as an asset extension to `metro.config.js`!
 * * If you are passing in a `{ url: ... }`, make sure the URL points directly to a `.onnx` model. This can either be a web URL (`http://..`/`https://..`), or a local file (`file://..`).
 *
 * @param source The `.onnx` model in form of either a `require(..)` statement or a `{ url: string }`.
 * @param provider The provider to use for computations. Uses the standard CPU provider per default. The `coreml` or `gpu` providers are hardware-accelerated.
 * @returns The state of the Model.
 */
export function useOnnxModel(
  source: ModelSource,
  provider: OnnxModelProvider = 'cpu'
): OnnxPlugin {
  const [state, setState] = useState<OnnxPlugin>({
    model: undefined,
    state: 'loading',
  })

  useEffect(() => {
    const load = async (): Promise<void> => {
      try {
        setState({ model: undefined, state: 'loading' })
        const m = await loadOnnxModel(source, provider)
        setState({ model: m, state: 'loaded' })
        console.log('ONNX Model loaded!')
      } catch (e) {
        console.error(`Failed to load ONNX Model ${source}!`, e)
        setState({ model: undefined, state: 'error', error: e as Error })
      }
    }
    load()
  }, [provider, source])

  return state
}