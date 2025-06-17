// include: shell.js
// The Module object: Our interface to the outside world. We import
// and export values on it. There are various ways Module can be used:
// 1. Not defined. We create it here
// 2. A function parameter, function(moduleArg) => Promise<Module>
// 3. pre-run appended it, var Module = {}; ..generated code..
// 4. External script tag defines var Module.
// We need to check if Module already exists (e.g. case 3 above).
// Substitution will be replaced with actual code on later stage of the build,
// this way Closure Compiler will not mangle it (e.g. case 4. above).
// Note that if you want to run closure, and also to use Module
// after the generated code, you will need to define   var Module = {};
// before the code. Then that object will be used in the code, and you
// can continue to use Module afterwards as well.
var Module = typeof Module != 'undefined' ? Module : {};

// Determine the runtime environment we are in. You can customize this by
// setting the ENVIRONMENT setting at compile time (see settings.js).

var ENVIRONMENT_IS_WEB = true;
var ENVIRONMENT_IS_WORKER = false;
var ENVIRONMENT_IS_NODE = false;
var ENVIRONMENT_IS_SHELL = false;

// --pre-jses are emitted after the Module integration code, so that they can
// refer to Module (if they choose; they can also define Module)


// Sometimes an existing Module object exists with properties
// meant to overwrite the default module functionality. Here
// we collect those properties and reapply _after_ we configure
// the current environment's defaults to avoid having to be so
// defensive during initialization.
var moduleOverrides = Object.assign({}, Module);

var arguments_ = [];
var thisProgram = './this.program';
var quit_ = (status, toThrow) => {
  throw toThrow;
};

// `/` should be present at the end if `scriptDirectory` is not empty
var scriptDirectory = '';
function locateFile(path) {
  if (Module['locateFile']) {
    return Module['locateFile'](path, scriptDirectory);
  }
  return scriptDirectory + path;
}

// Hooks that are implemented differently in different runtime environments.
var readAsync, readBinary;

// Note that this includes Node.js workers when relevant (pthreads is enabled).
// Node.js workers are detected as a combination of ENVIRONMENT_IS_WORKER and
// ENVIRONMENT_IS_NODE.
if (ENVIRONMENT_IS_WEB || ENVIRONMENT_IS_WORKER) {
  if (ENVIRONMENT_IS_WORKER) { // Check worker, not web, since window could be polyfilled
    scriptDirectory = self.location.href;
  } else if (typeof document != 'undefined' && document.currentScript) { // web
    scriptDirectory = document.currentScript.src;
  }
  // blob urls look like blob:http://site.com/etc/etc and we cannot infer anything from them.
  // otherwise, slice off the final part of the url to find the script directory.
  // if scriptDirectory does not contain a slash, lastIndexOf will return -1,
  // and scriptDirectory will correctly be replaced with an empty string.
  // If scriptDirectory contains a query (starting with ?) or a fragment (starting with #),
  // they are removed because they could contain a slash.
  if (scriptDirectory.startsWith('blob:')) {
    scriptDirectory = '';
  } else {
    scriptDirectory = scriptDirectory.slice(0, scriptDirectory.replace(/[?#].*/, '').lastIndexOf('/')+1);
  }

  {
// include: web_or_worker_shell_read.js
readAsync = async (url) => {
    var response = await fetch(url, { credentials: 'same-origin' });
    if (response.ok) {
      return response.arrayBuffer();
    }
    throw new Error(response.status + ' : ' + response.url);
  };
// end include: web_or_worker_shell_read.js
  }
} else
{
}

var out = Module['print'] || console.log.bind(console);
var err = Module['printErr'] || console.error.bind(console);

// Merge back in the overrides
Object.assign(Module, moduleOverrides);
// Free the object hierarchy contained in the overrides, this lets the GC
// reclaim data used.
moduleOverrides = null;

// Emit code to handle expected values on the Module object. This applies Module.x
// to the proper local x. This has two benefits: first, we only emit it if it is
// expected to arrive, and second, by using a local everywhere else that can be
// minified.

if (Module['arguments']) arguments_ = Module['arguments'];

if (Module['thisProgram']) thisProgram = Module['thisProgram'];

// perform assertions in shell.js after we set up out() and err(), as otherwise if an assertion fails it cannot print the message
// end include: shell.js

// include: preamble.js
// === Preamble library stuff ===

// Documentation for the public APIs defined in this file must be updated in:
//    site/source/docs/api_reference/preamble.js.rst
// A prebuilt local version of the documentation is available at:
//    site/build/text/docs/api_reference/preamble.js.txt
// You can also build docs locally as HTML or other formats in site/
// An online HTML version (which may be of a different version of Emscripten)
//    is up at http://kripken.github.io/emscripten-site/docs/api_reference/preamble.js.html

var wasmBinary = Module['wasmBinary'];

// Wasm globals

var wasmMemory;

//========================================
// Runtime essentials
//========================================

// whether we are quitting the application. no code should run after this.
// set in exit() and abort()
var ABORT = false;

// set by exit() and abort().  Passed to 'onExit' handler.
// NOTE: This is also used as the process return code code in shell environments
// but only when noExitRuntime is false.
var EXITSTATUS;

// In STRICT mode, we only define assert() when ASSERTIONS is set.  i.e. we
// don't define it at all in release modes.  This matches the behaviour of
// MINIMAL_RUNTIME.
// TODO(sbc): Make this the default even without STRICT enabled.
/** @type {function(*, string=)} */
function assert(condition, text) {
  if (!condition) {
    // This build was created without ASSERTIONS defined.  `assert()` should not
    // ever be called in this configuration but in case there are callers in
    // the wild leave this simple abort() implementation here for now.
    abort(text);
  }
}

// Memory management

var HEAP,
/** @type {!Int8Array} */
  HEAP8,
/** @type {!Uint8Array} */
  HEAPU8,
/** @type {!Int16Array} */
  HEAP16,
/** @type {!Uint16Array} */
  HEAPU16,
/** @type {!Int32Array} */
  HEAP32,
/** @type {!Uint32Array} */
  HEAPU32,
/** @type {!Float32Array} */
  HEAPF32,
/* BigInt64Array type is not correctly defined in closure
/** not-@type {!BigInt64Array} */
  HEAP64,
/* BigUint64Array type is not correctly defined in closure
/** not-t@type {!BigUint64Array} */
  HEAPU64,
/** @type {!Float64Array} */
  HEAPF64;

var runtimeInitialized = false;

/**
 * Indicates whether filename is delivered via file protocol (as opposed to http/https)
 * @noinline
 */
var isFileURI = (filename) => filename.startsWith('file://');

// include: runtime_shared.js
// include: runtime_stack_check.js
// end include: runtime_stack_check.js
// include: runtime_exceptions.js
// end include: runtime_exceptions.js
// include: runtime_debug.js
// end include: runtime_debug.js
// include: memoryprofiler.js
// end include: memoryprofiler.js


function updateMemoryViews() {
  var b = wasmMemory.buffer;
  Module['HEAP8'] = HEAP8 = new Int8Array(b);
  Module['HEAP16'] = HEAP16 = new Int16Array(b);
  Module['HEAPU8'] = HEAPU8 = new Uint8Array(b);
  Module['HEAPU16'] = HEAPU16 = new Uint16Array(b);
  Module['HEAP32'] = HEAP32 = new Int32Array(b);
  Module['HEAPU32'] = HEAPU32 = new Uint32Array(b);
  Module['HEAPF32'] = HEAPF32 = new Float32Array(b);
  Module['HEAPF64'] = HEAPF64 = new Float64Array(b);
  Module['HEAP64'] = HEAP64 = new BigInt64Array(b);
  Module['HEAPU64'] = HEAPU64 = new BigUint64Array(b);
}

// end include: runtime_shared.js
function preRun() {
  if (Module['preRun']) {
    if (typeof Module['preRun'] == 'function') Module['preRun'] = [Module['preRun']];
    while (Module['preRun'].length) {
      addOnPreRun(Module['preRun'].shift());
    }
  }
  callRuntimeCallbacks(onPreRuns);
}

function initRuntime() {
  runtimeInitialized = true;

  

  wasmExports['__wasm_call_ctors']();

  
}

function postRun() {

  if (Module['postRun']) {
    if (typeof Module['postRun'] == 'function') Module['postRun'] = [Module['postRun']];
    while (Module['postRun'].length) {
      addOnPostRun(Module['postRun'].shift());
    }
  }

  callRuntimeCallbacks(onPostRuns);
}

// A counter of dependencies for calling run(). If we need to
// do asynchronous work before running, increment this and
// decrement it. Incrementing must happen in a place like
// Module.preRun (used by emcc to add file preloading).
// Note that you can add dependencies in preRun, even though
// it happens right before run - run will be postponed until
// the dependencies are met.
var runDependencies = 0;
var dependenciesFulfilled = null; // overridden to take different actions when all run dependencies are fulfilled

function getUniqueRunDependency(id) {
  return id;
}

function addRunDependency(id) {
  runDependencies++;

  Module['monitorRunDependencies']?.(runDependencies);

}

function removeRunDependency(id) {
  runDependencies--;

  Module['monitorRunDependencies']?.(runDependencies);

  if (runDependencies == 0) {
    if (dependenciesFulfilled) {
      var callback = dependenciesFulfilled;
      dependenciesFulfilled = null;
      callback(); // can add another dependenciesFulfilled
    }
  }
}

/** @param {string|number=} what */
function abort(what) {
  Module['onAbort']?.(what);

  what = 'Aborted(' + what + ')';
  // TODO(sbc): Should we remove printing and leave it up to whoever
  // catches the exception?
  err(what);

  ABORT = true;

  what += '. Build with -sASSERTIONS for more info.';

  // Use a wasm runtime error, because a JS error might be seen as a foreign
  // exception, which means we'd run destructors on it. We need the error to
  // simply make the program stop.
  // FIXME This approach does not work in Wasm EH because it currently does not assume
  // all RuntimeErrors are from traps; it decides whether a RuntimeError is from
  // a trap or not based on a hidden field within the object. So at the moment
  // we don't have a way of throwing a wasm trap from JS. TODO Make a JS API that
  // allows this in the wasm spec.

  // Suppress closure compiler warning here. Closure compiler's builtin extern
  // definition for WebAssembly.RuntimeError claims it takes no arguments even
  // though it can.
  // TODO(https://github.com/google/closure-compiler/pull/3913): Remove if/when upstream closure gets fixed.
  /** @suppress {checkTypes} */
  var e = new WebAssembly.RuntimeError(what);

  // Throw the error whether or not MODULARIZE is set because abort is used
  // in code paths apart from instantiation where an exception is expected
  // to be thrown when abort is called.
  throw e;
}

var wasmBinaryFile;
function findWasmBinary() {
    return locateFile('minigpu_web.wasm');
}

function getBinarySync(file) {
  if (file == wasmBinaryFile && wasmBinary) {
    return new Uint8Array(wasmBinary);
  }
  if (readBinary) {
    return readBinary(file);
  }
  throw 'both async and sync fetching of the wasm failed';
}

async function getWasmBinary(binaryFile) {
  // If we don't have the binary yet, load it asynchronously using readAsync.
  if (!wasmBinary) {
    // Fetch the binary using readAsync
    try {
      var response = await readAsync(binaryFile);
      return new Uint8Array(response);
    } catch {
      // Fall back to getBinarySync below;
    }
  }

  // Otherwise, getBinarySync should be able to get it synchronously
  return getBinarySync(binaryFile);
}

async function instantiateArrayBuffer(binaryFile, imports) {
  try {
    var binary = await getWasmBinary(binaryFile);
    var instance = await WebAssembly.instantiate(binary, imports);
    return instance;
  } catch (reason) {
    err(`failed to asynchronously prepare wasm: ${reason}`);

    abort(reason);
  }
}

async function instantiateAsync(binary, binaryFile, imports) {
  if (!binary && typeof WebAssembly.instantiateStreaming == 'function'
     ) {
    try {
      var response = fetch(binaryFile, { credentials: 'same-origin' });
      var instantiationResult = await WebAssembly.instantiateStreaming(response, imports);
      return instantiationResult;
    } catch (reason) {
      // We expect the most common failure cause to be a bad MIME type for the binary,
      // in which case falling back to ArrayBuffer instantiation should work.
      err(`wasm streaming compile failed: ${reason}`);
      err('falling back to ArrayBuffer instantiation');
      // fall back of instantiateArrayBuffer below
    };
  }
  return instantiateArrayBuffer(binaryFile, imports);
}

function getWasmImports() {
  // prepare imports
  return {
    'env': wasmImports,
    'wasi_snapshot_preview1': wasmImports,
  }
}

// Create the wasm instance.
// Receives the wasm imports, returns the exports.
async function createWasm() {
  // Load the wasm module and create an instance of using native support in the JS engine.
  // handle a generated wasm instance, receiving its exports and
  // performing other necessary setup
  /** @param {WebAssembly.Module=} module*/
  function receiveInstance(instance, module) {
    wasmExports = instance.exports;

    wasmExports = Asyncify.instrumentWasmExports(wasmExports);

    

    wasmMemory = wasmExports['memory'];
    
    updateMemoryViews();

    removeRunDependency('wasm-instantiate');
    return wasmExports;
  }
  // wait for the pthread pool (if any)
  addRunDependency('wasm-instantiate');

  // Prefer streaming instantiation if available.
  function receiveInstantiationResult(result) {
    // 'result' is a ResultObject object which has both the module and instance.
    // receiveInstance() will swap in the exports (to Module.asm) so they can be called
    // TODO: Due to Closure regression https://github.com/google/closure-compiler/issues/3193, the above line no longer optimizes out down to the following line.
    // When the regression is fixed, can restore the above PTHREADS-enabled path.
    return receiveInstance(result['instance']);
  }

  var info = getWasmImports();

  // User shell pages can write their own Module.instantiateWasm = function(imports, successCallback) callback
  // to manually instantiate the Wasm module themselves. This allows pages to
  // run the instantiation parallel to any other async startup actions they are
  // performing.
  // Also pthreads and wasm workers initialize the wasm instance through this
  // path.
  if (Module['instantiateWasm']) {
    return new Promise((resolve, reject) => {
        Module['instantiateWasm'](info, (mod, inst) => {
          receiveInstance(mod, inst);
          resolve(mod.exports);
        });
    });
  }

  wasmBinaryFile ??= findWasmBinary();

    var result = await instantiateAsync(wasmBinary, wasmBinaryFile, info);
    var exports = receiveInstantiationResult(result);
    return exports;
}

// === Body ===
// end include: preamble.js


  class ExitStatus {
      name = 'ExitStatus';
      constructor(status) {
        this.message = `Program terminated with exit(${status})`;
        this.status = status;
      }
    }

  var callRuntimeCallbacks = (callbacks) => {
      while (callbacks.length > 0) {
        // Pass the module as the first argument.
        callbacks.shift()(Module);
      }
    };
  var onPostRuns = [];
  var addOnPostRun = (cb) => onPostRuns.unshift(cb);

  var onPreRuns = [];
  var addOnPreRun = (cb) => onPreRuns.unshift(cb);


  
    /**
     * @param {number} ptr
     * @param {string} type
     */
  function getValue(ptr, type = 'i8') {
    if (type.endsWith('*')) type = '*';
    switch (type) {
      case 'i1': return HEAP8[ptr];
      case 'i8': return HEAP8[ptr];
      case 'i16': return HEAP16[((ptr)>>1)];
      case 'i32': return HEAP32[((ptr)>>2)];
      case 'i64': return HEAP64[((ptr)>>3)];
      case 'float': return HEAPF32[((ptr)>>2)];
      case 'double': return HEAPF64[((ptr)>>3)];
      case '*': return HEAPU32[((ptr)>>2)];
      default: abort(`invalid type for getValue: ${type}`);
    }
  }

  var noExitRuntime = Module['noExitRuntime'] || true;

  
    /**
     * @param {number} ptr
     * @param {number} value
     * @param {string} type
     */
  function setValue(ptr, value, type = 'i8') {
    if (type.endsWith('*')) type = '*';
    switch (type) {
      case 'i1': HEAP8[ptr] = value; break;
      case 'i8': HEAP8[ptr] = value; break;
      case 'i16': HEAP16[((ptr)>>1)] = value; break;
      case 'i32': HEAP32[((ptr)>>2)] = value; break;
      case 'i64': HEAP64[((ptr)>>3)] = BigInt(value); break;
      case 'float': HEAPF32[((ptr)>>2)] = value; break;
      case 'double': HEAPF64[((ptr)>>3)] = value; break;
      case '*': HEAPU32[((ptr)>>2)] = value; break;
      default: abort(`invalid type for setValue: ${type}`);
    }
  }

  var stackRestore = (val) => __emscripten_stack_restore(val);

  var stackSave = () => _emscripten_stack_get_current();

  var UTF8Decoder = new TextDecoder();
  
    /**
     * Given a pointer 'ptr' to a null-terminated UTF8-encoded string in the
     * emscripten HEAP, returns a copy of that string as a Javascript String object.
     *
     * @param {number} ptr
     * @param {number=} maxBytesToRead - An optional length that specifies the
     *   maximum number of bytes to read. You can omit this parameter to scan the
     *   string until the first 0 byte. If maxBytesToRead is passed, and the string
     *   at [ptr, ptr+maxBytesToReadr[ contains a null byte in the middle, then the
     *   string will cut short at that byte index (i.e. maxBytesToRead will not
     *   produce a string of exact length [ptr, ptr+maxBytesToRead[) N.B. mixing
     *   frequent uses of UTF8ToString() with and without maxBytesToRead may throw
     *   JS JIT optimizations off, so it is worth to consider consistently using one
     * @return {string}
     */
  var UTF8ToString = (ptr, maxBytesToRead) => {
      if (!ptr) return '';
      var maxPtr = ptr + maxBytesToRead;
      for (var end = ptr; !(end >= maxPtr) && HEAPU8[end];) ++end;
      return UTF8Decoder.decode(HEAPU8.subarray(ptr, end));
    };
  var ___assert_fail = (condition, filename, line, func) =>
      abort(`Assertion failed: ${UTF8ToString(condition)}, at: ` + [filename ? UTF8ToString(filename) : 'unknown filename', line, func ? UTF8ToString(func) : 'unknown function']);

  class ExceptionInfo {
      // excPtr - Thrown object pointer to wrap. Metadata pointer is calculated from it.
      constructor(excPtr) {
        this.excPtr = excPtr;
        this.ptr = excPtr - 24;
      }
  
      set_type(type) {
        HEAPU32[(((this.ptr)+(4))>>2)] = type;
      }
  
      get_type() {
        return HEAPU32[(((this.ptr)+(4))>>2)];
      }
  
      set_destructor(destructor) {
        HEAPU32[(((this.ptr)+(8))>>2)] = destructor;
      }
  
      get_destructor() {
        return HEAPU32[(((this.ptr)+(8))>>2)];
      }
  
      set_caught(caught) {
        caught = caught ? 1 : 0;
        HEAP8[(this.ptr)+(12)] = caught;
      }
  
      get_caught() {
        return HEAP8[(this.ptr)+(12)] != 0;
      }
  
      set_rethrown(rethrown) {
        rethrown = rethrown ? 1 : 0;
        HEAP8[(this.ptr)+(13)] = rethrown;
      }
  
      get_rethrown() {
        return HEAP8[(this.ptr)+(13)] != 0;
      }
  
      // Initialize native structure fields. Should be called once after allocated.
      init(type, destructor) {
        this.set_adjusted_ptr(0);
        this.set_type(type);
        this.set_destructor(destructor);
      }
  
      set_adjusted_ptr(adjustedPtr) {
        HEAPU32[(((this.ptr)+(16))>>2)] = adjustedPtr;
      }
  
      get_adjusted_ptr() {
        return HEAPU32[(((this.ptr)+(16))>>2)];
      }
    }
  
  var exceptionLast = 0;
  
  var uncaughtExceptionCount = 0;
  var ___cxa_throw = (ptr, type, destructor) => {
      var info = new ExceptionInfo(ptr);
      // Initialize ExceptionInfo content after it was allocated in __cxa_allocate_exception.
      info.init(type, destructor);
      exceptionLast = ptr;
      uncaughtExceptionCount++;
      throw exceptionLast;
    };

  var __abort_js = () =>
      abort('');

  var runtimeKeepaliveCounter = 0;
  var __emscripten_runtime_keepalive_clear = () => {
      noExitRuntime = false;
      runtimeKeepaliveCounter = 0;
    };

  var isLeapYear = (year) => year%4 === 0 && (year%100 !== 0 || year%400 === 0);
  
  var MONTH_DAYS_LEAP_CUMULATIVE = [0,31,60,91,121,152,182,213,244,274,305,335];
  
  var MONTH_DAYS_REGULAR_CUMULATIVE = [0,31,59,90,120,151,181,212,243,273,304,334];
  var ydayFromDate = (date) => {
      var leap = isLeapYear(date.getFullYear());
      var monthDaysCumulative = (leap ? MONTH_DAYS_LEAP_CUMULATIVE : MONTH_DAYS_REGULAR_CUMULATIVE);
      var yday = monthDaysCumulative[date.getMonth()] + date.getDate() - 1; // -1 since it's days since Jan 1
  
      return yday;
    };
  
  var INT53_MAX = 9007199254740992;
  
  var INT53_MIN = -9007199254740992;
  var bigintToI53Checked = (num) => (num < INT53_MIN || num > INT53_MAX) ? NaN : Number(num);
  function __localtime_js(time, tmPtr) {
    time = bigintToI53Checked(time);
  
    
      var date = new Date(time*1000);
      HEAP32[((tmPtr)>>2)] = date.getSeconds();
      HEAP32[(((tmPtr)+(4))>>2)] = date.getMinutes();
      HEAP32[(((tmPtr)+(8))>>2)] = date.getHours();
      HEAP32[(((tmPtr)+(12))>>2)] = date.getDate();
      HEAP32[(((tmPtr)+(16))>>2)] = date.getMonth();
      HEAP32[(((tmPtr)+(20))>>2)] = date.getFullYear()-1900;
      HEAP32[(((tmPtr)+(24))>>2)] = date.getDay();
  
      var yday = ydayFromDate(date)|0;
      HEAP32[(((tmPtr)+(28))>>2)] = yday;
      HEAP32[(((tmPtr)+(36))>>2)] = -(date.getTimezoneOffset() * 60);
  
      // Attention: DST is in December in South, and some regions don't have DST at all.
      var start = new Date(date.getFullYear(), 0, 1);
      var summerOffset = new Date(date.getFullYear(), 6, 1).getTimezoneOffset();
      var winterOffset = start.getTimezoneOffset();
      var dst = (summerOffset != winterOffset && date.getTimezoneOffset() == Math.min(winterOffset, summerOffset))|0;
      HEAP32[(((tmPtr)+(32))>>2)] = dst;
    ;
  }

  var timers = {
  };
  
  var handleException = (e) => {
      // Certain exception types we do not treat as errors since they are used for
      // internal control flow.
      // 1. ExitStatus, which is thrown by exit()
      // 2. "unwind", which is thrown by emscripten_unwind_to_js_event_loop() and others
      //    that wish to return to JS event loop.
      if (e instanceof ExitStatus || e == 'unwind') {
        return EXITSTATUS;
      }
      quit_(1, e);
    };
  
  
  var keepRuntimeAlive = () => noExitRuntime || runtimeKeepaliveCounter > 0;
  var _proc_exit = (code) => {
      EXITSTATUS = code;
      if (!keepRuntimeAlive()) {
        Module['onExit']?.(code);
        ABORT = true;
      }
      quit_(code, new ExitStatus(code));
    };
  /** @suppress {duplicate } */
  /** @param {boolean|number=} implicit */
  var exitJS = (status, implicit) => {
      EXITSTATUS = status;
  
      _proc_exit(status);
    };
  var _exit = exitJS;
  
  
  var maybeExit = () => {
      if (!keepRuntimeAlive()) {
        try {
          _exit(EXITSTATUS);
        } catch (e) {
          handleException(e);
        }
      }
    };
  var callUserCallback = (func) => {
      if (ABORT) {
        return;
      }
      try {
        func();
        maybeExit();
      } catch (e) {
        handleException(e);
      }
    };
  
  
  var _emscripten_get_now = () => performance.now();
  var __setitimer_js = (which, timeout_ms) => {
      // First, clear any existing timer.
      if (timers[which]) {
        clearTimeout(timers[which].id);
        delete timers[which];
      }
  
      // A timeout of zero simply cancels the current timeout so we have nothing
      // more to do.
      if (!timeout_ms) return 0;
  
      var id = setTimeout(() => {
        delete timers[which];
        callUserCallback(() => __emscripten_timeout(which, _emscripten_get_now()));
      }, timeout_ms);
      timers[which] = { id, timeout_ms };
      return 0;
    };

  var stringToUTF8Array = (str, heap, outIdx, maxBytesToWrite) => {
      // Parameter maxBytesToWrite is not optional. Negative values, 0, null,
      // undefined and false each don't write out any bytes.
      if (!(maxBytesToWrite > 0))
        return 0;
  
      var startIdx = outIdx;
      var endIdx = outIdx + maxBytesToWrite - 1; // -1 for string null terminator.
      for (var i = 0; i < str.length; ++i) {
        // Gotcha: charCodeAt returns a 16-bit word that is a UTF-16 encoded code
        // unit, not a Unicode code point of the character! So decode
        // UTF16->UTF32->UTF8.
        // See http://unicode.org/faq/utf_bom.html#utf16-3
        // For UTF8 byte structure, see http://en.wikipedia.org/wiki/UTF-8#Description
        // and https://www.ietf.org/rfc/rfc2279.txt
        // and https://tools.ietf.org/html/rfc3629
        var u = str.charCodeAt(i); // possibly a lead surrogate
        if (u >= 0xD800 && u <= 0xDFFF) {
          var u1 = str.charCodeAt(++i);
          u = 0x10000 + ((u & 0x3FF) << 10) | (u1 & 0x3FF);
        }
        if (u <= 0x7F) {
          if (outIdx >= endIdx) break;
          heap[outIdx++] = u;
        } else if (u <= 0x7FF) {
          if (outIdx + 1 >= endIdx) break;
          heap[outIdx++] = 0xC0 | (u >> 6);
          heap[outIdx++] = 0x80 | (u & 63);
        } else if (u <= 0xFFFF) {
          if (outIdx + 2 >= endIdx) break;
          heap[outIdx++] = 0xE0 | (u >> 12);
          heap[outIdx++] = 0x80 | ((u >> 6) & 63);
          heap[outIdx++] = 0x80 | (u & 63);
        } else {
          if (outIdx + 3 >= endIdx) break;
          heap[outIdx++] = 0xF0 | (u >> 18);
          heap[outIdx++] = 0x80 | ((u >> 12) & 63);
          heap[outIdx++] = 0x80 | ((u >> 6) & 63);
          heap[outIdx++] = 0x80 | (u & 63);
        }
      }
      // Null-terminate the pointer to the buffer.
      heap[outIdx] = 0;
      return outIdx - startIdx;
    };
  var stringToUTF8 = (str, outPtr, maxBytesToWrite) => {
      return stringToUTF8Array(str, HEAPU8, outPtr, maxBytesToWrite);
    };
  var __tzset_js = (timezone, daylight, std_name, dst_name) => {
      // TODO: Use (malleable) environment variables instead of system settings.
      var currentYear = new Date().getFullYear();
      var winter = new Date(currentYear, 0, 1);
      var summer = new Date(currentYear, 6, 1);
      var winterOffset = winter.getTimezoneOffset();
      var summerOffset = summer.getTimezoneOffset();
  
      // Local standard timezone offset. Local standard time is not adjusted for
      // daylight savings.  This code uses the fact that getTimezoneOffset returns
      // a greater value during Standard Time versus Daylight Saving Time (DST).
      // Thus it determines the expected output during Standard Time, and it
      // compares whether the output of the given date the same (Standard) or less
      // (DST).
      var stdTimezoneOffset = Math.max(winterOffset, summerOffset);
  
      // timezone is specified as seconds west of UTC ("The external variable
      // `timezone` shall be set to the difference, in seconds, between
      // Coordinated Universal Time (UTC) and local standard time."), the same
      // as returned by stdTimezoneOffset.
      // See http://pubs.opengroup.org/onlinepubs/009695399/functions/tzset.html
      HEAPU32[((timezone)>>2)] = stdTimezoneOffset * 60;
  
      HEAP32[((daylight)>>2)] = Number(winterOffset != summerOffset);
  
      var extractZone = (timezoneOffset) => {
        // Why inverse sign?
        // Read here https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/getTimezoneOffset
        var sign = timezoneOffset >= 0 ? "-" : "+";
  
        var absOffset = Math.abs(timezoneOffset)
        var hours = String(Math.floor(absOffset / 60)).padStart(2, "0");
        var minutes = String(absOffset % 60).padStart(2, "0");
  
        return `UTC${sign}${hours}${minutes}`;
      }
  
      var winterName = extractZone(winterOffset);
      var summerName = extractZone(summerOffset);
      if (summerOffset < winterOffset) {
        // Northern hemisphere
        stringToUTF8(winterName, std_name, 17);
        stringToUTF8(summerName, dst_name, 17);
      } else {
        stringToUTF8(winterName, dst_name, 17);
        stringToUTF8(summerName, std_name, 17);
      }
    };

  
  var _emscripten_date_now = () => Date.now();
  
  var nowIsMonotonic = 1;
  
  var checkWasiClock = (clock_id) => clock_id >= 0 && clock_id <= 3;
  
  function _clock_time_get(clk_id, ignored_precision, ptime) {
    ignored_precision = bigintToI53Checked(ignored_precision);
  
    
      if (!checkWasiClock(clk_id)) {
        return 28;
      }
      var now;
      // all wasi clocks but realtime are monotonic
      if (clk_id === 0) {
        now = _emscripten_date_now();
      } else if (nowIsMonotonic) {
        now = _emscripten_get_now();
      } else {
        return 52;
      }
      // "now" is in ms, and wasi times are in ns.
      var nsec = Math.round(now * 1000 * 1000);
      HEAP64[((ptime)>>3)] = BigInt(nsec);
      return 0;
    ;
  }

  var _emscripten_has_asyncify = () => 1;

  var getHeapMax = () =>
      // Stay one Wasm page short of 4GB: while e.g. Chrome is able to allocate
      // full 4GB Wasm memories, the size will wrap back to 0 bytes in Wasm side
      // for any code that deals with heap sizes, which would require special
      // casing all heap size related code to treat 0 specially.
      2147483648;
  
  var alignMemory = (size, alignment) => {
      return Math.ceil(size / alignment) * alignment;
    };
  
  var growMemory = (size) => {
      var b = wasmMemory.buffer;
      var pages = ((size - b.byteLength + 65535) / 65536) | 0;
      try {
        // round size grow request up to wasm page size (fixed 64KB per spec)
        wasmMemory.grow(pages); // .grow() takes a delta compared to the previous size
        updateMemoryViews();
        return 1 /*success*/;
      } catch(e) {
      }
      // implicit 0 return to save code size (caller will cast "undefined" into 0
      // anyhow)
    };
  var _emscripten_resize_heap = (requestedSize) => {
      var oldSize = HEAPU8.length;
      // With CAN_ADDRESS_2GB or MEMORY64, pointers are already unsigned.
      requestedSize >>>= 0;
      // With multithreaded builds, races can happen (another thread might increase the size
      // in between), so return a failure, and let the caller retry.
  
      // Memory resize rules:
      // 1.  Always increase heap size to at least the requested size, rounded up
      //     to next page multiple.
      // 2a. If MEMORY_GROWTH_LINEAR_STEP == -1, excessively resize the heap
      //     geometrically: increase the heap size according to
      //     MEMORY_GROWTH_GEOMETRIC_STEP factor (default +20%), At most
      //     overreserve by MEMORY_GROWTH_GEOMETRIC_CAP bytes (default 96MB).
      // 2b. If MEMORY_GROWTH_LINEAR_STEP != -1, excessively resize the heap
      //     linearly: increase the heap size by at least
      //     MEMORY_GROWTH_LINEAR_STEP bytes.
      // 3.  Max size for the heap is capped at 2048MB-WASM_PAGE_SIZE, or by
      //     MAXIMUM_MEMORY, or by ASAN limit, depending on which is smallest
      // 4.  If we were unable to allocate as much memory, it may be due to
      //     over-eager decision to excessively reserve due to (3) above.
      //     Hence if an allocation fails, cut down on the amount of excess
      //     growth, in an attempt to succeed to perform a smaller allocation.
  
      // A limit is set for how much we can grow. We should not exceed that
      // (the wasm binary specifies it, so if we tried, we'd fail anyhow).
      var maxHeapSize = getHeapMax();
      if (requestedSize > maxHeapSize) {
        return false;
      }
  
      // Loop through potential heap size increases. If we attempt a too eager
      // reservation that fails, cut down on the attempted size and reserve a
      // smaller bump instead. (max 3 times, chosen somewhat arbitrarily)
      for (var cutDown = 1; cutDown <= 4; cutDown *= 2) {
        var overGrownHeapSize = oldSize * (1 + 0.2 / cutDown); // ensure geometric growth
        // but limit overreserving (default to capping at +96MB overgrowth at most)
        overGrownHeapSize = Math.min(overGrownHeapSize, requestedSize + 100663296 );
  
        var newSize = Math.min(maxHeapSize, alignMemory(Math.max(requestedSize, overGrownHeapSize), 65536));
  
        var replacement = growMemory(newSize);
        if (replacement) {
  
          return true;
        }
      }
      return false;
    };

  /** @param {number=} timeout */
  var safeSetTimeout = (func, timeout) => {
      
      return setTimeout(() => {
        
        callUserCallback(func);
      }, timeout);
    };
  var _emscripten_sleep = (ms) => {
      // emscripten_sleep() does not return a value, but we still need a |return|
      // here for stack switching support (ASYNCIFY=2). In that mode this function
      // returns a Promise instead of nothing, and that Promise is what tells the
      // wasm VM to pause the stack.
      return Asyncify.handleSleep((wakeUp) => safeSetTimeout(wakeUp, ms));
    };
  _emscripten_sleep.isAsync = true;

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  var lengthBytesUTF8 = (str) => {
      var len = 0;
      for (var i = 0; i < str.length; ++i) {
        // Gotcha: charCodeAt returns a 16-bit word that is a UTF-16 encoded code
        // unit, not a Unicode code point of the character! So decode
        // UTF16->UTF32->UTF8.
        // See http://unicode.org/faq/utf_bom.html#utf16-3
        var c = str.charCodeAt(i); // possibly a lead surrogate
        if (c <= 0x7F) {
          len++;
        } else if (c <= 0x7FF) {
          len += 2;
        } else if (c >= 0xD800 && c <= 0xDFFF) {
          len += 4; ++i;
        } else {
          len += 3;
        }
      }
      return len;
    };
  
  
  var stackAlloc = (sz) => __emscripten_stack_alloc(sz);
  var stringToUTF8OnStack = (str) => {
      var size = lengthBytesUTF8(str) + 1;
      var ret = stackAlloc(size);
      stringToUTF8(str, ret, size);
      return ret;
    };
  
  
  
  
  var WebGPU = {
  Internals:{
  jsObjects:[],
  jsObjectInsert:(ptr, jsObject) => {
          WebGPU.Internals.jsObjects[ptr] = jsObject;
        },
  bufferOnUnmaps:[],
  futures:[],
  futureInsert:(futureId, promise) => {
          WebGPU.Internals.futures[futureId] =
            new Promise((resolve) => promise.finally(() => resolve(futureId)));
        },
  },
  getJsObject:(ptr) => {
        if (!ptr) return undefined;
        return WebGPU.Internals.jsObjects[ptr];
      },
  importJsAdapter:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateAdapter(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsBindGroup:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateBindGroup(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsBindGroupLayout:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateBindGroupLayout(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsBuffer:(buffer, parentPtr = 0) => {
        // At the moment, we do not allow importing pending buffers.
        assert(buffer.mapState != "pending");
        var mapState = buffer.mapState == "mapped" ?
          3 :
          1;
        var bufferPtr = _emwgpuCreateBuffer(parentPtr, mapState);
        WebGPU.Internals.jsObjectInsert(bufferPtr, buffer);
        if (buffer.mapState == "mapped") {
          WebGPU.Internals.bufferOnUnmaps[bufferPtr] = [];
        }
        return bufferPtr;
      },
  importJsCommandBuffer:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateCommandBuffer(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsCommandEncoder:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateCommandEncoder(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsComputePassEncoder:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateComputePassEncoder(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsComputePipeline:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateComputePipeline(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsDevice:(device, parentPtr = 0) => {
        var queuePtr = _emwgpuCreateQueue(parentPtr);
        var devicePtr = _emwgpuCreateDevice(parentPtr, queuePtr);
        WebGPU.Internals.jsObjectInsert(queuePtr, device.queue);
        WebGPU.Internals.jsObjectInsert(devicePtr, device);
        return devicePtr;
      },
  importJsPipelineLayout:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreatePipelineLayout(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsQuerySet:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateQuerySet(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsQueue:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateQueue(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsRenderBundle:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateRenderBundle(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsRenderBundleEncoder:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateRenderBundleEncoder(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsRenderPassEncoder:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateRenderPassEncoder(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsRenderPipeline:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateRenderPipeline(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsSampler:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateSampler(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsShaderModule:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateShaderModule(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsSurface:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateSurface(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsTexture:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateTexture(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  importJsTextureView:(obj, parentPtr = 0) => {
            var ptr = _emwgpuCreateTextureView(parentPtr);
            WebGPU.Internals.jsObjects[ptr] = obj;
            return ptr;
          },
  errorCallback:(callback, type, message, userdata) => {
        var sp = stackSave();
        var messagePtr = stringToUTF8OnStack(message);
        ((a1, a2, a3) => {} /* a dynamic function call to signature viii, but there are no exported function pointers with that signature, so this path should never be taken. Build with ASSERTIONS enabled to validate. */)(type, messagePtr, userdata);
        stackRestore(sp);
      },
  setStringView:(ptr, data, length) => {
        HEAPU32[((ptr)>>2)] = data;
        HEAPU32[(((ptr)+(4))>>2)] = length;
      },
  makeStringFromStringView:(stringViewPtr) => {
        var ptr = HEAPU32[((stringViewPtr)>>2)];
        var length = HEAPU32[(((stringViewPtr)+(4))>>2)];
        // UTF8ToString stops at the first null terminator character in the
        // string regardless of the length.
        return UTF8ToString(ptr, length);
      },
  makeStringFromOptionalStringView:(stringViewPtr) => {
        var ptr = HEAPU32[((stringViewPtr)>>2)];
        var length = HEAPU32[(((stringViewPtr)+(4))>>2)];
        // If we don't have a valid string pointer, just return undefined when
        // optional.
        if (!ptr) {
          if (length === 0) {
            return "";
          }
          return undefined;
        }
        // UTF8ToString stops at the first null terminator character in the
        // string regardless of the length.
        return UTF8ToString(ptr, length);
      },
  makeColor:(ptr) => {
        return {
          "r": HEAPF64[((ptr)>>3)],
          "g": HEAPF64[(((ptr)+(8))>>3)],
          "b": HEAPF64[(((ptr)+(16))>>3)],
          "a": HEAPF64[(((ptr)+(24))>>3)],
        };
      },
  makeExtent3D:(ptr) => {
        return {
          "width": HEAPU32[((ptr)>>2)],
          "height": HEAPU32[(((ptr)+(4))>>2)],
          "depthOrArrayLayers": HEAPU32[(((ptr)+(8))>>2)],
        };
      },
  makeOrigin3D:(ptr) => {
        return {
          "x": HEAPU32[((ptr)>>2)],
          "y": HEAPU32[(((ptr)+(4))>>2)],
          "z": HEAPU32[(((ptr)+(8))>>2)],
        };
      },
  makeTexelCopyTextureInfo:(ptr) => {
        
        return {
          "texture": WebGPU.getJsObject(
            HEAPU32[((ptr)>>2)]),
          "mipLevel": HEAPU32[(((ptr)+(4))>>2)],
          "origin": WebGPU.makeOrigin3D(ptr + 8),
          "aspect": WebGPU.TextureAspect[HEAPU32[(((ptr)+(20))>>2)]],
        };
      },
  makeTexelCopyBufferLayout:(ptr) => {
        var bytesPerRow = HEAPU32[(((ptr)+(8))>>2)];
        var rowsPerImage = HEAPU32[(((ptr)+(12))>>2)];
        return {
          "offset": (HEAPU32[(((ptr + 4))>>2)] * 0x100000000 + HEAPU32[((ptr)>>2)]),
          "bytesPerRow": bytesPerRow === 4294967295 ? undefined : bytesPerRow,
          "rowsPerImage": rowsPerImage === 4294967295 ? undefined : rowsPerImage,
        };
      },
  makeTexelCopyBufferInfo:(ptr) => {
        
        var layoutPtr = ptr + 0;
        var bufferCopyView = WebGPU.makeTexelCopyBufferLayout(layoutPtr);
        bufferCopyView["buffer"] = WebGPU.getJsObject(
          HEAPU32[(((ptr)+(16))>>2)]);
        return bufferCopyView;
      },
  makePassTimestampWrites:(ptr) => {
        if (ptr === 0) return undefined;
        return {
          "querySet": WebGPU.getJsObject(
            HEAPU32[(((ptr)+(4))>>2)]),
          "beginningOfPassWriteIndex": HEAPU32[(((ptr)+(8))>>2)],
          "endOfPassWriteIndex": HEAPU32[(((ptr)+(12))>>2)],
        };
      },
  makePipelineConstants:(constantCount, constantsPtr) => {
        if (!constantCount) return;
        var constants = {};
        for (var i = 0; i < constantCount; ++i) {
          var entryPtr = constantsPtr + 24 * i;
          var key = WebGPU.makeStringFromStringView(entryPtr + 4);
          constants[key] = HEAPF64[(((entryPtr)+(16))>>3)];
        }
        return constants;
      },
  makePipelineLayout:(layoutPtr) => {
        if (!layoutPtr) return 'auto';
        return WebGPU.getJsObject(layoutPtr);
      },
  makeComputeState:(ptr) => {
        if (!ptr) return undefined;
        
        var desc = {
          "module": WebGPU.getJsObject(
            HEAPU32[(((ptr)+(4))>>2)]),
          "constants": WebGPU.makePipelineConstants(
            HEAPU32[(((ptr)+(16))>>2)],
            HEAPU32[(((ptr)+(20))>>2)]),
          "entryPoint": WebGPU.makeStringFromOptionalStringView(
            ptr + 8),
        };
        return desc;
      },
  makeComputePipelineDesc:(descriptor) => {
        
  
        var desc = {
          "label": WebGPU.makeStringFromOptionalStringView(
            descriptor + 4),
          "layout": WebGPU.makePipelineLayout(
            HEAPU32[(((descriptor)+(12))>>2)]),
          "compute": WebGPU.makeComputeState(
            descriptor + 16),
        };
        return desc;
      },
  makeRenderPipelineDesc:(descriptor) => {
        
  
        function makePrimitiveState(psPtr) {
          if (!psPtr) return undefined;
          
          return {
            "topology": WebGPU.PrimitiveTopology[
              HEAPU32[(((psPtr)+(4))>>2)]],
            "stripIndexFormat": WebGPU.IndexFormat[
              HEAPU32[(((psPtr)+(8))>>2)]],
            "frontFace": WebGPU.FrontFace[
              HEAPU32[(((psPtr)+(12))>>2)]],
            "cullMode": WebGPU.CullMode[
              HEAPU32[(((psPtr)+(16))>>2)]],
            "unclippedDepth":
              !!(HEAPU32[(((psPtr)+(20))>>2)]),
          };
        }
  
        function makeBlendComponent(bdPtr) {
          if (!bdPtr) return undefined;
          return {
            "operation": WebGPU.BlendOperation[
              HEAPU32[((bdPtr)>>2)]],
            "srcFactor": WebGPU.BlendFactor[
              HEAPU32[(((bdPtr)+(4))>>2)]],
            "dstFactor": WebGPU.BlendFactor[
              HEAPU32[(((bdPtr)+(8))>>2)]],
          };
        }
  
        function makeBlendState(bsPtr) {
          if (!bsPtr) return undefined;
          return {
            "alpha": makeBlendComponent(bsPtr + 12),
            "color": makeBlendComponent(bsPtr + 0),
          };
        }
  
        function makeColorState(csPtr) {
          
          var formatInt = HEAPU32[(((csPtr)+(4))>>2)];
          return formatInt === 0 ? undefined : {
            "format": WebGPU.TextureFormat[formatInt],
            "blend": makeBlendState(HEAPU32[(((csPtr)+(8))>>2)]),
            "writeMask": HEAPU32[(((csPtr)+(16))>>2)],
          };
        }
  
        function makeColorStates(count, csArrayPtr) {
          var states = [];
          for (var i = 0; i < count; ++i) {
            states.push(makeColorState(csArrayPtr + 24 * i));
          }
          return states;
        }
  
        function makeStencilStateFace(ssfPtr) {
          
          return {
            "compare": WebGPU.CompareFunction[
              HEAPU32[((ssfPtr)>>2)]],
            "failOp": WebGPU.StencilOperation[
              HEAPU32[(((ssfPtr)+(4))>>2)]],
            "depthFailOp": WebGPU.StencilOperation[
              HEAPU32[(((ssfPtr)+(8))>>2)]],
            "passOp": WebGPU.StencilOperation[
              HEAPU32[(((ssfPtr)+(12))>>2)]],
          };
        }
  
        function makeDepthStencilState(dssPtr) {
          if (!dssPtr) return undefined;
  
          
          return {
            "format": WebGPU.TextureFormat[
              HEAPU32[(((dssPtr)+(4))>>2)]],
            "depthWriteEnabled": !!(HEAPU32[(((dssPtr)+(8))>>2)]),
            "depthCompare": WebGPU.CompareFunction[
              HEAPU32[(((dssPtr)+(12))>>2)]],
            "stencilFront": makeStencilStateFace(dssPtr + 16),
            "stencilBack": makeStencilStateFace(dssPtr + 32),
            "stencilReadMask": HEAPU32[(((dssPtr)+(48))>>2)],
            "stencilWriteMask": HEAPU32[(((dssPtr)+(52))>>2)],
            "depthBias": HEAP32[(((dssPtr)+(56))>>2)],
            "depthBiasSlopeScale": HEAPF32[(((dssPtr)+(60))>>2)],
            "depthBiasClamp": HEAPF32[(((dssPtr)+(64))>>2)],
          };
        }
  
        function makeVertexAttribute(vaPtr) {
          
          return {
            "format": WebGPU.VertexFormat[
              HEAPU32[(((vaPtr)+(4))>>2)]],
            "offset": (HEAPU32[((((vaPtr + 4))+(8))>>2)] * 0x100000000 + HEAPU32[(((vaPtr)+(8))>>2)]),
            "shaderLocation": HEAPU32[(((vaPtr)+(16))>>2)],
          };
        }
  
        function makeVertexAttributes(count, vaArrayPtr) {
          var vas = [];
          for (var i = 0; i < count; ++i) {
            vas.push(makeVertexAttribute(vaArrayPtr + i * 24));
          }
          return vas;
        }
  
        function makeVertexBuffer(vbPtr) {
          if (!vbPtr) return undefined;
          var stepModeInt = HEAPU32[(((vbPtr)+(4))>>2)];
          var attributeCountInt = HEAPU32[(((vbPtr)+(16))>>2)];
          if (stepModeInt === 0 && attributeCountInt === 0) {
            return null;
          }
          return {
            "arrayStride": (HEAPU32[((((vbPtr + 4))+(8))>>2)] * 0x100000000 + HEAPU32[(((vbPtr)+(8))>>2)]),
            "stepMode": WebGPU.VertexStepMode[stepModeInt],
            "attributes": makeVertexAttributes(
              attributeCountInt,
              HEAPU32[(((vbPtr)+(20))>>2)]),
          };
        }
  
        function makeVertexBuffers(count, vbArrayPtr) {
          if (!count) return undefined;
  
          var vbs = [];
          for (var i = 0; i < count; ++i) {
            vbs.push(makeVertexBuffer(vbArrayPtr + i * 24));
          }
          return vbs;
        }
  
        function makeVertexState(viPtr) {
          if (!viPtr) return undefined;
          
          var desc = {
            "module": WebGPU.getJsObject(
              HEAPU32[(((viPtr)+(4))>>2)]),
            "constants": WebGPU.makePipelineConstants(
              HEAPU32[(((viPtr)+(16))>>2)],
              HEAPU32[(((viPtr)+(20))>>2)]),
            "buffers": makeVertexBuffers(
              HEAPU32[(((viPtr)+(24))>>2)],
              HEAPU32[(((viPtr)+(28))>>2)]),
            "entryPoint": WebGPU.makeStringFromOptionalStringView(
              viPtr + 8),
            };
          return desc;
        }
  
        function makeMultisampleState(msPtr) {
          if (!msPtr) return undefined;
          
          return {
            "count": HEAPU32[(((msPtr)+(4))>>2)],
            "mask": HEAPU32[(((msPtr)+(8))>>2)],
            "alphaToCoverageEnabled": !!(HEAPU32[(((msPtr)+(12))>>2)]),
          };
        }
  
        function makeFragmentState(fsPtr) {
          if (!fsPtr) return undefined;
          
          var desc = {
            "module": WebGPU.getJsObject(
              HEAPU32[(((fsPtr)+(4))>>2)]),
            "constants": WebGPU.makePipelineConstants(
              HEAPU32[(((fsPtr)+(16))>>2)],
              HEAPU32[(((fsPtr)+(20))>>2)]),
            "targets": makeColorStates(
              HEAPU32[(((fsPtr)+(24))>>2)],
              HEAPU32[(((fsPtr)+(28))>>2)]),
            "entryPoint": WebGPU.makeStringFromOptionalStringView(
              fsPtr + 8),
            };
          return desc;
        }
  
        var desc = {
          "label": WebGPU.makeStringFromOptionalStringView(
            descriptor + 4),
          "layout": WebGPU.makePipelineLayout(
            HEAPU32[(((descriptor)+(12))>>2)]),
          "vertex": makeVertexState(
            descriptor + 16),
          "primitive": makePrimitiveState(
            descriptor + 48),
          "depthStencil": makeDepthStencilState(
            HEAPU32[(((descriptor)+(72))>>2)]),
          "multisample": makeMultisampleState(
            descriptor + 76),
          "fragment": makeFragmentState(
            HEAPU32[(((descriptor)+(92))>>2)]),
        };
        return desc;
      },
  fillLimitStruct:(limits, limitsOutPtr) => {
        
  
        function setLimitValueU32(name, limitOffset) {
          var limitValue = limits[name];
          HEAP32[(((limitsOutPtr)+(limitOffset))>>2)] = limitValue;
        }
        function setLimitValueU64(name, limitOffset) {
          var limitValue = limits[name];
          HEAP64[(((limitsOutPtr)+(limitOffset))>>3)] = BigInt(limitValue);
        }
  
        setLimitValueU32('maxTextureDimension1D', 4);
        setLimitValueU32('maxTextureDimension2D', 8);
        setLimitValueU32('maxTextureDimension3D', 12);
        setLimitValueU32('maxTextureArrayLayers', 16);
        setLimitValueU32('maxBindGroups', 20);
        setLimitValueU32('maxBindGroupsPlusVertexBuffers', 24);
        setLimitValueU32('maxBindingsPerBindGroup', 28);
        setLimitValueU32('maxDynamicUniformBuffersPerPipelineLayout', 32);
        setLimitValueU32('maxDynamicStorageBuffersPerPipelineLayout', 36);
        setLimitValueU32('maxSampledTexturesPerShaderStage', 40);
        setLimitValueU32('maxSamplersPerShaderStage', 44);
        setLimitValueU32('maxStorageBuffersPerShaderStage', 48);
        setLimitValueU32('maxStorageTexturesPerShaderStage', 52);
        setLimitValueU32('maxUniformBuffersPerShaderStage', 56);
        setLimitValueU32('minUniformBufferOffsetAlignment', 80);
        setLimitValueU32('minStorageBufferOffsetAlignment', 84);
  
        setLimitValueU64('maxUniformBufferBindingSize', 64);
        setLimitValueU64('maxStorageBufferBindingSize', 72);
  
        setLimitValueU32('maxVertexBuffers', 88);
        setLimitValueU64('maxBufferSize', 96);
        setLimitValueU32('maxVertexAttributes', 104);
        setLimitValueU32('maxVertexBufferArrayStride', 108);
        setLimitValueU32('maxInterStageShaderVariables', 112);
        setLimitValueU32('maxColorAttachments', 116);
        setLimitValueU32('maxColorAttachmentBytesPerSample', 120);
        setLimitValueU32('maxComputeWorkgroupStorageSize', 124);
        setLimitValueU32('maxComputeInvocationsPerWorkgroup', 128);
        setLimitValueU32('maxComputeWorkgroupSizeX', 132);
        setLimitValueU32('maxComputeWorkgroupSizeY', 136);
        setLimitValueU32('maxComputeWorkgroupSizeZ', 140);
        setLimitValueU32('maxComputeWorkgroupsPerDimension', 144);
      },
  Int_BufferMapState:{
  unmapped:1,
  pending:2,
  mapped:3,
  },
  Int_CompilationMessageType:{
  error:1,
  warning:2,
  info:3,
  },
  Int_DeviceLostReason:{
  undefined:1,
  unknown:1,
  destroyed:2,
  },
  Int_PreferredFormat:{
  rgba8unorm:18,
  bgra8unorm:23,
  },
  WGSLLanguageFeatureName:{
  1:"readonly_and_readwrite_storage_textures",
  2:"packed_4x8_integer_dot_product",
  3:"unrestricted_pointer_parameters",
  4:"pointer_composite_access",
  5:"sized_binding_array",
  },
  AddressMode:[,"clamp-to-edge","repeat","mirror-repeat"],
  BlendFactor:[,"zero","one","src","one-minus-src","src-alpha","one-minus-src-alpha","dst","one-minus-dst","dst-alpha","one-minus-dst-alpha","src-alpha-saturated","constant","one-minus-constant","src1","one-minus-src1","src1alpha","one-minus-src1alpha"],
  BlendOperation:[,"add","subtract","reverse-subtract","min","max"],
  BufferBindingType:["binding-not-used",,"uniform","storage","read-only-storage"],
  BufferMapState:{
  1:"unmapped",
  2:"pending",
  3:"mapped",
  },
  CompareFunction:[,"never","less","equal","less-equal","greater","not-equal","greater-equal","always"],
  CompilationInfoRequestStatus:{
  1:"success",
  2:"instance-dropped",
  },
  CompositeAlphaMode:[,"opaque","premultiplied","unpremultiplied","inherit"],
  CullMode:[,"none","front","back"],
  ErrorFilter:{
  1:"validation",
  2:"out-of-memory",
  3:"internal",
  },
  FeatureLevel:[,"compatibility","core"],
  FeatureName:{
  1:"depth-clip-control",
  2:"depth32float-stencil8",
  3:"timestamp-query",
  4:"texture-compression-bc",
  5:"texture-compression-bc-sliced-3d",
  6:"texture-compression-etc2",
  7:"texture-compression-astc",
  8:"texture-compression-astc-sliced-3d",
  9:"indirect-first-instance",
  10:"shader-f16",
  11:"rg11b10ufloat-renderable",
  12:"bgra8unorm-storage",
  13:"float32-filterable",
  14:"float32-blendable",
  15:"clip-distances",
  16:"dual-source-blending",
  17:"subgroups",
  18:"core-features-and-limits",
  327688:"subgroups-f16",
  327693:"chromium-experimental-unorm16-texture-formats",
  327694:"chromium-experimental-snorm16-texture-formats",
  327733:"chromium-experimental-multi-draw-indirect",
  },
  FilterMode:[,"nearest","linear"],
  FrontFace:[,"ccw","cw"],
  IndexFormat:[,"uint16","uint32"],
  LoadOp:[,"load","clear"],
  MipmapFilterMode:[,"nearest","linear"],
  OptionalBool:["false","true",],
  PowerPreference:[,"low-power","high-performance"],
  PredefinedColorSpace:{
  1:"srgb",
  2:"display-p3",
  },
  PrimitiveTopology:[,"point-list","line-list","line-strip","triangle-list","triangle-strip"],
  QueryType:{
  1:"occlusion",
  2:"timestamp",
  },
  SamplerBindingType:["binding-not-used",,"filtering","non-filtering","comparison"],
  Status:{
  1:"success",
  2:"error",
  },
  StencilOperation:[,"keep","zero","replace","invert","increment-clamp","decrement-clamp","increment-wrap","decrement-wrap"],
  StorageTextureAccess:["binding-not-used",,"write-only","read-only","read-write"],
  StoreOp:[,"store","discard"],
  SurfaceGetCurrentTextureStatus:{
  1:"success-optimal",
  2:"success-suboptimal",
  3:"timeout",
  4:"outdated",
  5:"lost",
  6:"error",
  },
  TextureAspect:[,"all","stencil-only","depth-only"],
  TextureDimension:[,"1d","2d","3d"],
  TextureFormat:[,"r8unorm","r8snorm","r8uint","r8sint","r16uint","r16sint","r16float","rg8unorm","rg8snorm","rg8uint","rg8sint","r32float","r32uint","r32sint","rg16uint","rg16sint","rg16float","rgba8unorm","rgba8unorm-srgb","rgba8snorm","rgba8uint","rgba8sint","bgra8unorm","bgra8unorm-srgb","rgb10a2uint","rgb10a2unorm","rg11b10ufloat","rgb9e5ufloat","rg32float","rg32uint","rg32sint","rgba16uint","rgba16sint","rgba16float","rgba32float","rgba32uint","rgba32sint","stencil8","depth16unorm","depth24plus","depth24plus-stencil8","depth32float","depth32float-stencil8","bc1-rgba-unorm","bc1-rgba-unorm-srgb","bc2-rgba-unorm","bc2-rgba-unorm-srgb","bc3-rgba-unorm","bc3-rgba-unorm-srgb","bc4-r-unorm","bc4-r-snorm","bc5-rg-unorm","bc5-rg-snorm","bc6h-rgb-ufloat","bc6h-rgb-float","bc7-rgba-unorm","bc7-rgba-unorm-srgb","etc2-rgb8unorm","etc2-rgb8unorm-srgb","etc2-rgb8a1unorm","etc2-rgb8a1unorm-srgb","etc2-rgba8unorm","etc2-rgba8unorm-srgb","eac-r11unorm","eac-r11snorm","eac-rg11unorm","eac-rg11snorm","astc-4x4-unorm","astc-4x4-unorm-srgb","astc-5x4-unorm","astc-5x4-unorm-srgb","astc-5x5-unorm","astc-5x5-unorm-srgb","astc-6x5-unorm","astc-6x5-unorm-srgb","astc-6x6-unorm","astc-6x6-unorm-srgb","astc-8x5-unorm","astc-8x5-unorm-srgb","astc-8x6-unorm","astc-8x6-unorm-srgb","astc-8x8-unorm","astc-8x8-unorm-srgb","astc-10x5-unorm","astc-10x5-unorm-srgb","astc-10x6-unorm","astc-10x6-unorm-srgb","astc-10x8-unorm","astc-10x8-unorm-srgb","astc-10x10-unorm","astc-10x10-unorm-srgb","astc-12x10-unorm","astc-12x10-unorm-srgb","astc-12x12-unorm","astc-12x12-unorm-srgb"],
  TextureSampleType:["binding-not-used",,"float","unfilterable-float","depth","sint","uint"],
  TextureViewDimension:[,"1d","2d","2d-array","cube","cube-array","3d"],
  ToneMappingMode:{
  1:"standard",
  2:"extended",
  },
  VertexFormat:{
  1:"uint8",
  2:"uint8x2",
  3:"uint8x4",
  4:"sint8",
  5:"sint8x2",
  6:"sint8x4",
  7:"unorm8",
  8:"unorm8x2",
  9:"unorm8x4",
  10:"snorm8",
  11:"snorm8x2",
  12:"snorm8x4",
  13:"uint16",
  14:"uint16x2",
  15:"uint16x4",
  16:"sint16",
  17:"sint16x2",
  18:"sint16x4",
  19:"unorm16",
  20:"unorm16x2",
  21:"unorm16x4",
  22:"snorm16",
  23:"snorm16x2",
  24:"snorm16x4",
  25:"float16",
  26:"float16x2",
  27:"float16x4",
  28:"float32",
  29:"float32x2",
  30:"float32x3",
  31:"float32x4",
  32:"uint32",
  33:"uint32x2",
  34:"uint32x3",
  35:"uint32x4",
  36:"sint32",
  37:"sint32x2",
  38:"sint32x3",
  39:"sint32x4",
  40:"unorm10-10-10-2",
  41:"unorm8x4-bgra",
  },
  VertexStepMode:[,"vertex","instance"],
  FeatureNameString2Enum:{
  'depth-clip-control':"1",
  'depth32float-stencil8':"2",
  'timestamp-query':"3",
  'texture-compression-bc':"4",
  'texture-compression-bc-sliced-3d':"5",
  'texture-compression-etc2':"6",
  'texture-compression-astc':"7",
  'texture-compression-astc-sliced-3d':"8",
  'indirect-first-instance':"9",
  'shader-f16':"10",
  'rg11b10ufloat-renderable':"11",
  'bgra8unorm-storage':"12",
  'float32-filterable':"13",
  'float32-blendable':"14",
  'clip-distances':"15",
  'dual-source-blending':"16",
  subgroups:"17",
  'core-features-and-limits':"18",
  'subgroups-f16':"327688",
  'chromium-experimental-unorm16-texture-formats':"327693",
  'chromium-experimental-snorm16-texture-formats':"327694",
  'chromium-experimental-multi-draw-indirect':"327733",
  },
  WGSLLanguageFeatureNameString2Enum:{
  readonly_and_readwrite_storage_textures:"1",
  packed_4x8_integer_dot_product:"2",
  unrestricted_pointer_parameters:"3",
  pointer_composite_access:"4",
  sized_binding_array:"5",
  },
  };
  
  function _emwgpuAdapterRequestDevice(adapterPtr, futureId, deviceLostFutureId, devicePtr, queuePtr, descriptor) {
    futureId = bigintToI53Checked(futureId);
    deviceLostFutureId = bigintToI53Checked(deviceLostFutureId);
  
    
      var adapter = WebGPU.getJsObject(adapterPtr);
  
      var desc = {};
      if (descriptor) {
        
        var requiredFeatureCount = HEAPU32[(((descriptor)+(12))>>2)];
        if (requiredFeatureCount) {
          var requiredFeaturesPtr = HEAPU32[(((descriptor)+(16))>>2)];
          // requiredFeaturesPtr is a pointer to an array of FeatureName which is an enum of size uint32_t
          desc["requiredFeatures"] = Array.from(HEAPU32.subarray((((requiredFeaturesPtr)>>2)), ((requiredFeaturesPtr + requiredFeatureCount * 4)>>2)),
            (feature) => WebGPU.FeatureName[feature]);
        }
        var limitsPtr = HEAPU32[(((descriptor)+(20))>>2)];
        if (limitsPtr) {
          
          var requiredLimits = {};
          function setLimitU32IfDefined(name, limitOffset) {
            var ptr = limitsPtr + limitOffset;
            var value = HEAPU32[((ptr)>>2)];
            if (value != 4294967295) {
              requiredLimits[name] = value;
            }
          }
          function setLimitU64IfDefined(name, limitOffset) {
            var ptr = limitsPtr + limitOffset;
            // Handle WGPU_LIMIT_U64_UNDEFINED.
            var limitPart1 = HEAPU32[((ptr)>>2)];
            var limitPart2 = HEAPU32[(((ptr)+(4))>>2)];
            if (limitPart1 != 0xFFFFFFFF || limitPart2 != 0xFFFFFFFF) {
              requiredLimits[name] = (HEAPU32[(((ptr + 4))>>2)] * 0x100000000 + HEAPU32[((ptr)>>2)])
            }
          }
  
          setLimitU32IfDefined("maxTextureDimension1D", 4);
          setLimitU32IfDefined("maxTextureDimension2D", 8);
          setLimitU32IfDefined("maxTextureDimension3D", 12);
          setLimitU32IfDefined("maxTextureArrayLayers", 16);
          setLimitU32IfDefined("maxBindGroups", 20);
          setLimitU32IfDefined('maxBindGroupsPlusVertexBuffers', 24);
          setLimitU32IfDefined("maxDynamicUniformBuffersPerPipelineLayout", 32);
          setLimitU32IfDefined("maxDynamicStorageBuffersPerPipelineLayout", 36);
          setLimitU32IfDefined("maxSampledTexturesPerShaderStage", 40);
          setLimitU32IfDefined("maxSamplersPerShaderStage", 44);
          setLimitU32IfDefined("maxStorageBuffersPerShaderStage", 48);
          setLimitU32IfDefined("maxStorageTexturesPerShaderStage", 52);
          setLimitU32IfDefined("maxUniformBuffersPerShaderStage", 56);
          setLimitU32IfDefined("minUniformBufferOffsetAlignment", 80);
          setLimitU32IfDefined("minStorageBufferOffsetAlignment", 84);
          setLimitU64IfDefined("maxUniformBufferBindingSize", 64);
          setLimitU64IfDefined("maxStorageBufferBindingSize", 72);
          setLimitU32IfDefined("maxVertexBuffers", 88);
          setLimitU64IfDefined("maxBufferSize", 96);
          setLimitU32IfDefined("maxVertexAttributes", 104);
          setLimitU32IfDefined("maxVertexBufferArrayStride", 108);
          setLimitU32IfDefined("maxInterStageShaderVariables", 112);
          setLimitU32IfDefined("maxColorAttachments", 116);
          setLimitU32IfDefined("maxColorAttachmentBytesPerSample", 120);
          setLimitU32IfDefined("maxComputeWorkgroupStorageSize", 124);
          setLimitU32IfDefined("maxComputeInvocationsPerWorkgroup", 128);
          setLimitU32IfDefined("maxComputeWorkgroupSizeX", 132);
          setLimitU32IfDefined("maxComputeWorkgroupSizeY", 136);
          setLimitU32IfDefined("maxComputeWorkgroupSizeZ", 140);
          setLimitU32IfDefined("maxComputeWorkgroupsPerDimension", 144);
          desc["requiredLimits"] = requiredLimits;
        }
  
        var defaultQueuePtr = HEAPU32[(((descriptor)+(24))>>2)];
        if (defaultQueuePtr) {
          var defaultQueueDesc = {
            "label": WebGPU.makeStringFromOptionalStringView(
              defaultQueuePtr + 4),
          };
          desc["defaultQueue"] = defaultQueueDesc;
        }
        desc["label"] = WebGPU.makeStringFromOptionalStringView(
          descriptor + 4
        );
      }
  
      
      WebGPU.Internals.futureInsert(futureId, adapter.requestDevice(desc).then((device) => {
        
        WebGPU.Internals.jsObjectInsert(queuePtr, device.queue);
        WebGPU.Internals.jsObjectInsert(devicePtr, device);
  
        
  
        // Set up device lost promise resolution.
        if (deviceLostFutureId) {
          
          WebGPU.Internals.futureInsert(deviceLostFutureId, device.lost.then((info) => {
            
            // Unset the uncaptured error handler.
            device.onuncapturederror = (ev) => {};
            var sp = stackSave();
            var messagePtr = stringToUTF8OnStack(info.message);
            _emwgpuOnDeviceLostCompleted(deviceLostFutureId, WebGPU.Int_DeviceLostReason[info.reason],
              messagePtr);
            stackRestore(sp);
          }));
        }
  
        // Set up uncaptured error handlers.
        device.onuncapturederror = (ev) => {
            var type = 5;
            if (ev.error instanceof GPUValidationError) type = 2;
            else if (ev.error instanceof GPUOutOfMemoryError) type = 3;
            else if (ev.error instanceof GPUInternalError) type = 4;
            var sp = stackSave();
            var messagePtr = stringToUTF8OnStack(ev.error.message);
            _emwgpuOnUncapturedError(devicePtr, type, messagePtr);
            stackRestore(sp);
        };
  
        _emwgpuOnRequestDeviceCompleted(futureId, 1,
          devicePtr, 0);
      }, (ex) => {
        
        var sp = stackSave();
        var messagePtr = stringToUTF8OnStack(ex.message);
        _emwgpuOnRequestDeviceCompleted(futureId, 3,
          devicePtr, messagePtr);
        if (deviceLostFutureId) {
          _emwgpuOnDeviceLostCompleted(deviceLostFutureId, 4,
            messagePtr);
        }
        stackRestore(sp);
      }));
    ;
  }

  var _emwgpuBufferDestroy = (bufferPtr) => {
      var buffer = WebGPU.getJsObject(bufferPtr);
      var onUnmap = WebGPU.Internals.bufferOnUnmaps[bufferPtr];
      if (onUnmap) {
        for (var i = 0; i < onUnmap.length; ++i) {
          onUnmap[i]();
        }
        delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];
      }
  
      buffer.destroy();
    };

  var warnOnce = (text) => {
      warnOnce.shown ||= {};
      if (!warnOnce.shown[text]) {
        warnOnce.shown[text] = 1;
        err(text);
      }
    };
  
  
  
  var _emwgpuBufferGetConstMappedRange = (bufferPtr, offset, size) => {
      var buffer = WebGPU.getJsObject(bufferPtr);
  
      if (size === 0) warnOnce('getMappedRange size=0 no longer means WGPU_WHOLE_MAP_SIZE');
  
      if (size == -1) size = undefined;
  
      var mapped;
      try {
        mapped = buffer.getMappedRange(offset, size);
      } catch (ex) {
        return 0;
      }
      var data = _memalign(16, mapped.byteLength);
      HEAPU8.set(new Uint8Array(mapped), data);
      WebGPU.Internals.bufferOnUnmaps[bufferPtr].push(() => _free(data));
      return data;
    };

  
  
  var _emwgpuBufferMapAsync = function(bufferPtr, futureId, mode, offset, size) {
    futureId = bigintToI53Checked(futureId);
    mode = bigintToI53Checked(mode);
  
    
      var buffer = WebGPU.getJsObject(bufferPtr);
      WebGPU.Internals.bufferOnUnmaps[bufferPtr] = [];
  
      if (size == -1) size = undefined;
  
      
      WebGPU.Internals.futureInsert(futureId, buffer.mapAsync(mode, offset, size).then(() => {
        
        _emwgpuOnMapAsyncCompleted(futureId, 1,
          0);
      }, (ex) => {
        
        var sp = stackSave();
        var messagePtr = stringToUTF8OnStack(ex.message);
        var status =
          ex.name === 'AbortError' ? 4 :
          ex.name === 'OperationError' ? 3 :
          0;
        
        _emwgpuOnMapAsyncCompleted(futureId, status, messagePtr);
        delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];
      }));
    ;
  };

  var _emwgpuBufferUnmap = (bufferPtr) => {
      var buffer = WebGPU.getJsObject(bufferPtr);
  
      var onUnmap = WebGPU.Internals.bufferOnUnmaps[bufferPtr];
      if (!onUnmap) {
        // Already unmapped
        return;
      }
  
      for (var i = 0; i < onUnmap.length; ++i) {
        onUnmap[i]();
      }
      delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]
  
      buffer.unmap();
    };

  var _emwgpuDelete = (ptr) => {
      delete WebGPU.Internals.jsObjects[ptr];
    };

  var _emwgpuDeviceCreateBuffer = (devicePtr, descriptor, bufferPtr) => {
      
  
      var mappedAtCreation = !!(HEAPU32[(((descriptor)+(32))>>2)]);
  
      var desc = {
        "label": WebGPU.makeStringFromOptionalStringView(
          descriptor + 4),
        "usage": HEAPU32[(((descriptor)+(16))>>2)],
        "size": (HEAPU32[((((descriptor + 4))+(24))>>2)] * 0x100000000 + HEAPU32[(((descriptor)+(24))>>2)]),
        "mappedAtCreation": mappedAtCreation,
      };
  
      var device = WebGPU.getJsObject(devicePtr);
      WebGPU.Internals.jsObjectInsert(bufferPtr, device.createBuffer(desc));
      if (mappedAtCreation) {
        WebGPU.Internals.bufferOnUnmaps[bufferPtr] = [];
      }
    };

  var _emwgpuDeviceCreateShaderModule = (devicePtr, descriptor, shaderModulePtr) => {
      
      var nextInChainPtr = HEAPU32[((descriptor)>>2)];
      var sType = HEAPU32[(((nextInChainPtr)+(4))>>2)];
  
      var desc = {
        "label": WebGPU.makeStringFromOptionalStringView(
          descriptor + 4),
        "code": "",
      };
  
      switch (sType) {
        case 2: {
          desc["code"] = WebGPU.makeStringFromStringView(
            nextInChainPtr + 8
          );
          break;
        }
      }
  
      var device = WebGPU.getJsObject(devicePtr);
      WebGPU.Internals.jsObjectInsert(shaderModulePtr, device.createShaderModule(desc));
    };

  var _emwgpuDeviceDestroy = (devicePtr) => {
      WebGPU.getJsObject(devicePtr).destroy()
    };

  
  
  function _emwgpuInstanceRequestAdapter(instancePtr, futureId, options, adapterPtr) {
    futureId = bigintToI53Checked(futureId);
  
    
      var opts;
      if (options) {
        
        var featureLevel = HEAPU32[(((options)+(4))>>2)];
        opts = {
          "featureLevel": WebGPU.FeatureLevel[featureLevel],
          "powerPreference": WebGPU.PowerPreference[
            HEAPU32[(((options)+(8))>>2)]],
          "forceFallbackAdapter":
            !!(HEAPU32[(((options)+(12))>>2)]),
        };
  
        var nextInChainPtr = HEAPU32[((options)>>2)];
        if (nextInChainPtr !== 0) {
          var sType = HEAPU32[(((nextInChainPtr)+(4))>>2)];
          var webxrOptions = nextInChainPtr;
          
          opts.xrCompatible = !!(HEAPU32[(((webxrOptions)+(8))>>2)]);
        }
      }
  
      if (!('gpu' in navigator)) {
        var sp = stackSave();
        var messagePtr = stringToUTF8OnStack('WebGPU not available on this browser (navigator.gpu is not available)');
        _emwgpuOnRequestAdapterCompleted(futureId, 3,
          adapterPtr, messagePtr);
        stackRestore(sp);
        return;
      }
  
      
      WebGPU.Internals.futureInsert(futureId, navigator["gpu"]["requestAdapter"](opts).then((adapter) => {
        
        if (adapter) {
          WebGPU.Internals.jsObjectInsert(adapterPtr, adapter);
          _emwgpuOnRequestAdapterCompleted(futureId, 1,
            adapterPtr, 0);
        } else {
          var sp = stackSave();
          var messagePtr = stringToUTF8OnStack('WebGPU not available on this browser (requestAdapter returned null)');
          _emwgpuOnRequestAdapterCompleted(futureId, 3,
            adapterPtr, messagePtr);
          stackRestore(sp);
        }
      }, (ex) => {
        
        var sp = stackSave();
        var messagePtr = stringToUTF8OnStack(ex.message);
        _emwgpuOnRequestAdapterCompleted(futureId, 4,
          adapterPtr, messagePtr);
        stackRestore(sp);
      }));
    ;
  }

  var ENV = {
  };
  
  var getExecutableName = () => thisProgram || './this.program';
  var getEnvStrings = () => {
      if (!getEnvStrings.strings) {
        // Default values.
        // Browser language detection #8751
        var lang = ((typeof navigator == 'object' && navigator.languages && navigator.languages[0]) || 'C').replace('-', '_') + '.UTF-8';
        var env = {
          'USER': 'web_user',
          'LOGNAME': 'web_user',
          'PATH': '/',
          'PWD': '/',
          'HOME': '/home/web_user',
          'LANG': lang,
          '_': getExecutableName()
        };
        // Apply the user-provided values, if any.
        for (var x in ENV) {
          // x is a key in ENV; if ENV[x] is undefined, that means it was
          // explicitly set to be so. We allow user code to do that to
          // force variables with default values to remain unset.
          if (ENV[x] === undefined) delete env[x];
          else env[x] = ENV[x];
        }
        var strings = [];
        for (var x in env) {
          strings.push(`${x}=${env[x]}`);
        }
        getEnvStrings.strings = strings;
      }
      return getEnvStrings.strings;
    };
  
  var stringToAscii = (str, buffer) => {
      for (var i = 0; i < str.length; ++i) {
        HEAP8[buffer++] = str.charCodeAt(i);
      }
      // Null-terminate the string
      HEAP8[buffer] = 0;
    };
  var _environ_get = (__environ, environ_buf) => {
      var bufSize = 0;
      getEnvStrings().forEach((string, i) => {
        var ptr = environ_buf + bufSize;
        HEAPU32[(((__environ)+(i*4))>>2)] = ptr;
        stringToAscii(string, ptr);
        bufSize += string.length + 1;
      });
      return 0;
    };

  var _environ_sizes_get = (penviron_count, penviron_buf_size) => {
      var strings = getEnvStrings();
      HEAPU32[((penviron_count)>>2)] = strings.length;
      var bufSize = 0;
      strings.forEach((string) => bufSize += string.length + 1);
      HEAPU32[((penviron_buf_size)>>2)] = bufSize;
      return 0;
    };

  var _fd_close = (fd) => {
      return 52;
    };

  var _fd_read = (fd, iov, iovcnt, pnum) => {
      return 52;
    };

  function _fd_seek(fd, offset, whence, newOffset) {
    offset = bigintToI53Checked(offset);
  
    
      return 70;
    ;
  }

  var printCharBuffers = [null,[],[]];
  
  
    /**
     * Given a pointer 'idx' to a null-terminated UTF8-encoded string in the given
     * array that contains uint8 values, returns a copy of that string as a
     * Javascript String object.
     * heapOrArray is either a regular array, or a JavaScript typed array view.
     * @param {number=} idx
     * @param {number=} maxBytesToRead
     * @return {string}
     */
  var UTF8ArrayToString = (heapOrArray, idx = 0, maxBytesToRead = NaN) => {
      var endIdx = idx + maxBytesToRead;
      var endPtr = idx;
      // TextDecoder needs to know the byte length in advance, it doesn't stop on
      // null terminator by itself.  Also, use the length info to avoid running tiny
      // strings through TextDecoder, since .subarray() allocates garbage.
      // (As a tiny code save trick, compare endPtr against endIdx using a negation,
      // so that undefined/NaN means Infinity)
      while (heapOrArray[endPtr] && !(endPtr >= endIdx)) ++endPtr;
  
      return UTF8Decoder.decode(heapOrArray.buffer ? heapOrArray.subarray(idx, endPtr) : new Uint8Array(heapOrArray.slice(idx, endPtr)));
    };
  var printChar = (stream, curr) => {
      var buffer = printCharBuffers[stream];
      if (curr === 0 || curr === 10) {
        (stream === 1 ? out : err)(UTF8ArrayToString(buffer));
        buffer.length = 0;
      } else {
        buffer.push(curr);
      }
    };
  
  var flush_NO_FILESYSTEM = () => {
      // flush anything remaining in the buffers during shutdown
      if (printCharBuffers[1].length) printChar(1, 10);
      if (printCharBuffers[2].length) printChar(2, 10);
    };
  
  
  var SYSCALLS = {
  varargs:undefined,
  getStr(ptr) {
        var ret = UTF8ToString(ptr);
        return ret;
      },
  };
  var _fd_write = (fd, iov, iovcnt, pnum) => {
      // hack to support printf in SYSCALLS_REQUIRE_FILESYSTEM=0
      var num = 0;
      for (var i = 0; i < iovcnt; i++) {
        var ptr = HEAPU32[((iov)>>2)];
        var len = HEAPU32[(((iov)+(4))>>2)];
        iov += 8;
        for (var j = 0; j < len; j++) {
          printChar(fd, HEAPU8[ptr+j]);
        }
        num += len;
      }
      HEAPU32[((pnum)>>2)] = num;
      return 0;
    };


  
  var _wgpuCommandEncoderBeginComputePass = (encoderPtr, descriptor) => {
      var desc;
  
      if (descriptor) {
        
        desc = {
          "label": WebGPU.makeStringFromOptionalStringView(
            descriptor + 4),
          "timestampWrites": WebGPU.makePassTimestampWrites(
            HEAPU32[(((descriptor)+(12))>>2)]),
        };
      }
      var commandEncoder = WebGPU.getJsObject(encoderPtr);
      var ptr = _emwgpuCreateComputePassEncoder(0);
      WebGPU.Internals.jsObjectInsert(ptr, commandEncoder.beginComputePass(desc));
      return ptr;
    };

  
  function _wgpuCommandEncoderCopyBufferToBuffer(encoderPtr, srcPtr, srcOffset, dstPtr, dstOffset, size) {
    srcOffset = bigintToI53Checked(srcOffset);
    dstOffset = bigintToI53Checked(dstOffset);
    size = bigintToI53Checked(size);
  
    
      var commandEncoder = WebGPU.getJsObject(encoderPtr);
      var src = WebGPU.getJsObject(srcPtr);
      var dst = WebGPU.getJsObject(dstPtr);
      commandEncoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size);
    ;
  }

  
  var _wgpuCommandEncoderFinish = (encoderPtr, descriptor) => {
      // TODO: Use the descriptor.
      var commandEncoder = WebGPU.getJsObject(encoderPtr);
      var ptr = _emwgpuCreateCommandBuffer(0);
      WebGPU.Internals.jsObjectInsert(ptr, commandEncoder.finish());
      return ptr;
    };

  var _wgpuComputePassEncoderDispatchWorkgroups = (passPtr, x, y, z) => {
      var pass = WebGPU.getJsObject(passPtr);
      pass.dispatchWorkgroups(x, y, z);
    };

  var _wgpuComputePassEncoderEnd = (passPtr) => {
      var pass = WebGPU.getJsObject(passPtr);
      pass.end();
    };

  var _wgpuComputePassEncoderSetBindGroup = (passPtr, groupIndex, groupPtr, dynamicOffsetCount, dynamicOffsetsPtr) => {
      var pass = WebGPU.getJsObject(passPtr);
      var group = WebGPU.getJsObject(groupPtr);
      if (dynamicOffsetCount == 0) {
        pass.setBindGroup(groupIndex, group);
      } else {
        var offsets = [];
        for (var i = 0; i < dynamicOffsetCount; i++, dynamicOffsetsPtr += 4) {
          offsets.push(HEAPU32[((dynamicOffsetsPtr)>>2)]);
        }
        pass.setBindGroup(groupIndex, group, offsets);
      }
    };

  var _wgpuComputePassEncoderSetPipeline = (passPtr, pipelinePtr) => {
      var pass = WebGPU.getJsObject(passPtr);
      var pipeline = WebGPU.getJsObject(pipelinePtr);
      pass.setPipeline(pipeline);
    };

  var readI53FromI64 = (ptr) => {
      return HEAPU32[((ptr)>>2)] + HEAP32[(((ptr)+(4))>>2)] * 4294967296;
    };
  
  
  var _wgpuDeviceCreateBindGroup = (devicePtr, descriptor) => {
      
  
      function makeEntry(entryPtr) {
        
  
        var bufferPtr = HEAPU32[(((entryPtr)+(8))>>2)];
        var samplerPtr = HEAPU32[(((entryPtr)+(32))>>2)];
        var textureViewPtr = HEAPU32[(((entryPtr)+(36))>>2)];
  
        var binding = HEAPU32[(((entryPtr)+(4))>>2)];
  
        if (bufferPtr) {
          var size = readI53FromI64((entryPtr)+(24));
          if (size == -1) size = undefined;
  
          return {
            "binding": binding,
            "resource": {
              "buffer": WebGPU.getJsObject(bufferPtr),
              "offset": (HEAPU32[((((entryPtr + 4))+(16))>>2)] * 0x100000000 + HEAPU32[(((entryPtr)+(16))>>2)]),
              "size": size
            },
          };
        } else if (samplerPtr) {
          return {
            "binding": binding,
            "resource": WebGPU.getJsObject(samplerPtr),
          };
        } else {
          return {
            "binding": binding,
            "resource": WebGPU.getJsObject(textureViewPtr),
          };
        }
      }
  
      function makeEntries(count, entriesPtrs) {
        var entries = [];
        for (var i = 0; i < count; ++i) {
          entries.push(makeEntry(entriesPtrs +
              40 * i));
        }
        return entries;
      }
  
      var desc = {
        "label": WebGPU.makeStringFromOptionalStringView(
          descriptor + 4),
        "layout": WebGPU.getJsObject(
          HEAPU32[(((descriptor)+(12))>>2)]),
        "entries": makeEntries(
          HEAPU32[(((descriptor)+(16))>>2)],
          HEAPU32[(((descriptor)+(20))>>2)]
        ),
      };
  
      var device = WebGPU.getJsObject(devicePtr);
      var ptr = _emwgpuCreateBindGroup(0);
      WebGPU.Internals.jsObjectInsert(ptr, device.createBindGroup(desc));
      return ptr;
    };

  
  var _wgpuDeviceCreateBindGroupLayout = (devicePtr, descriptor) => {
      
  
      function makeBufferEntry(entryPtr) {
        
  
        var typeInt =
          HEAPU32[(((entryPtr)+(4))>>2)];
        if (!typeInt) return undefined;
  
        return {
          "type": WebGPU.BufferBindingType[typeInt],
          "hasDynamicOffset":
            !!(HEAPU32[(((entryPtr)+(8))>>2)]),
          "minBindingSize":
            (HEAPU32[((((entryPtr + 4))+(16))>>2)] * 0x100000000 + HEAPU32[(((entryPtr)+(16))>>2)]),
        };
      }
  
      function makeSamplerEntry(entryPtr) {
        
  
        var typeInt =
          HEAPU32[(((entryPtr)+(4))>>2)];
        if (!typeInt) return undefined;
  
        return {
          "type": WebGPU.SamplerBindingType[typeInt],
        };
      }
  
      function makeTextureEntry(entryPtr) {
        
  
        var sampleTypeInt =
          HEAPU32[(((entryPtr)+(4))>>2)];
        if (!sampleTypeInt) return undefined;
  
        return {
          "sampleType": WebGPU.TextureSampleType[sampleTypeInt],
          "viewDimension": WebGPU.TextureViewDimension[
            HEAPU32[(((entryPtr)+(8))>>2)]],
          "multisampled":
            !!(HEAPU32[(((entryPtr)+(12))>>2)]),
        };
      }
  
      function makeStorageTextureEntry(entryPtr) {
        
  
        var accessInt =
          HEAPU32[(((entryPtr)+(4))>>2)]
        if (!accessInt) return undefined;
  
        return {
          "access": WebGPU.StorageTextureAccess[accessInt],
          "format": WebGPU.TextureFormat[
            HEAPU32[(((entryPtr)+(8))>>2)]],
          "viewDimension": WebGPU.TextureViewDimension[
            HEAPU32[(((entryPtr)+(12))>>2)]],
        };
      }
  
      function makeEntry(entryPtr) {
        
  
        return {
          "binding":
            HEAPU32[(((entryPtr)+(4))>>2)],
          "visibility":
            HEAPU32[(((entryPtr)+(8))>>2)],
          "buffer": makeBufferEntry(entryPtr + 16),
          "sampler": makeSamplerEntry(entryPtr + 40),
          "texture": makeTextureEntry(entryPtr + 48),
          "storageTexture": makeStorageTextureEntry(entryPtr + 64),
        };
      }
  
      function makeEntries(count, entriesPtrs) {
        var entries = [];
        for (var i = 0; i < count; ++i) {
          entries.push(makeEntry(entriesPtrs +
              80 * i));
        }
        return entries;
      }
  
      var desc = {
        "label": WebGPU.makeStringFromOptionalStringView(
          descriptor + 4),
        "entries": makeEntries(
          HEAPU32[(((descriptor)+(12))>>2)],
          HEAPU32[(((descriptor)+(16))>>2)]
        ),
      };
  
      var device = WebGPU.getJsObject(devicePtr);
      var ptr = _emwgpuCreateBindGroupLayout(0);
      WebGPU.Internals.jsObjectInsert(ptr, device.createBindGroupLayout(desc));
      return ptr;
    };

  
  var _wgpuDeviceCreateCommandEncoder = (devicePtr, descriptor) => {
      var desc;
      if (descriptor) {
        
        desc = {
          "label": WebGPU.makeStringFromOptionalStringView(
            descriptor + 4),
        };
      }
      var device = WebGPU.getJsObject(devicePtr);
      var ptr = _emwgpuCreateCommandEncoder(0);
      WebGPU.Internals.jsObjectInsert(ptr, device.createCommandEncoder(desc));
      return ptr;
    };

  
  var _wgpuDeviceCreateComputePipeline = (devicePtr, descriptor) => {
      var desc = WebGPU.makeComputePipelineDesc(descriptor);
      var device = WebGPU.getJsObject(devicePtr);
      var ptr = _emwgpuCreateComputePipeline(0);
      WebGPU.Internals.jsObjectInsert(ptr, device.createComputePipeline(desc));
      return ptr;
    };

  
  var _wgpuDeviceCreatePipelineLayout = (devicePtr, descriptor) => {
      
      var bglCount = HEAPU32[(((descriptor)+(12))>>2)];
      var bglPtr = HEAPU32[(((descriptor)+(16))>>2)];
      var bgls = [];
      for (var i = 0; i < bglCount; ++i) {
        bgls.push(WebGPU.getJsObject(
          HEAPU32[(((bglPtr)+(4 * i))>>2)]));
      }
      var desc = {
        "label": WebGPU.makeStringFromOptionalStringView(
          descriptor + 4),
        "bindGroupLayouts": bgls,
      };
  
      var device = WebGPU.getJsObject(devicePtr);
      var ptr = _emwgpuCreatePipelineLayout(0);
      WebGPU.Internals.jsObjectInsert(ptr, device.createPipelineLayout(desc));
      return ptr;
    };

  var _wgpuQueueSubmit = (queuePtr, commandCount, commands) => {
      var queue = WebGPU.getJsObject(queuePtr);
      var cmds = Array.from(HEAP32.subarray((((commands)>>2)), ((commands + commandCount * 4)>>2)),
        (id) => WebGPU.getJsObject(id));
      queue.submit(cmds);
    };

  
  function _wgpuQueueWriteBuffer(queuePtr, bufferPtr, bufferOffset, data, size) {
    bufferOffset = bigintToI53Checked(bufferOffset);
  
    
      var queue = WebGPU.getJsObject(queuePtr);
      var buffer = WebGPU.getJsObject(bufferPtr);
      // There is a size limitation for ArrayBufferView. Work around by passing in a subarray
      // instead of the whole heap. crbug.com/1201109
      var subarray = HEAPU8.subarray(data, data + size);
      queue.writeBuffer(buffer, bufferOffset, subarray, 0, size);
    ;
  }

  var ptrToString = (ptr) => {
      // With CAN_ADDRESS_2GB or MEMORY64, pointers are already unsigned.
      ptr >>>= 0;
      return '0x' + ptr.toString(16).padStart(8, '0');
    };

  var runAndAbortIfError = (func) => {
      try {
        return func();
      } catch (e) {
        abort(e);
      }
    };
  
  
  var sigToWasmTypes = (sig) => {
      var typeNames = {
        'i': 'i32',
        'j': 'i64',
        'f': 'f32',
        'd': 'f64',
        'e': 'externref',
        'p': 'i32',
      };
      var type = {
        parameters: [],
        results: sig[0] == 'v' ? [] : [typeNames[sig[0]]]
      };
      for (var i = 1; i < sig.length; ++i) {
        type.parameters.push(typeNames[sig[i]]);
      }
      return type;
    };
  
  var runtimeKeepalivePush = () => {
      runtimeKeepaliveCounter += 1;
    };
  
  var runtimeKeepalivePop = () => {
      runtimeKeepaliveCounter -= 1;
    };
  
  
  var Asyncify = {
  instrumentWasmImports(imports) {
        var importPattern = /^(__asyncjs__.*)$/;
  
        for (let [x, original] of Object.entries(imports)) {
          if (typeof original == 'function') {
            let isAsyncifyImport = original.isAsync || importPattern.test(x);
          }
        }
      },
  instrumentWasmExports(exports) {
        var ret = {};
        for (let [x, original] of Object.entries(exports)) {
          if (typeof original == 'function') {
            ret[x] = (...args) => {
              Asyncify.exportCallStack.push(x);
              try {
                return original(...args);
              } finally {
                if (!ABORT) {
                  var y = Asyncify.exportCallStack.pop();
                  Asyncify.maybeStopUnwind();
                }
              }
            };
          } else {
            ret[x] = original;
          }
        }
        return ret;
      },
  State:{
  Normal:0,
  Unwinding:1,
  Rewinding:2,
  Disabled:3,
  },
  state:0,
  StackSize:4096,
  currData:null,
  handleSleepReturnValue:0,
  exportCallStack:[],
  callStackNameToId:{
  },
  callStackIdToName:{
  },
  callStackId:0,
  asyncPromiseHandlers:null,
  sleepCallbacks:[],
  getCallStackId(funcName) {
        var id = Asyncify.callStackNameToId[funcName];
        if (id === undefined) {
          id = Asyncify.callStackId++;
          Asyncify.callStackNameToId[funcName] = id;
          Asyncify.callStackIdToName[id] = funcName;
        }
        return id;
      },
  maybeStopUnwind() {
        if (Asyncify.currData &&
            Asyncify.state === Asyncify.State.Unwinding &&
            Asyncify.exportCallStack.length === 0) {
          // We just finished unwinding.
          // Be sure to set the state before calling any other functions to avoid
          // possible infinite recursion here (For example in debug pthread builds
          // the dbg() function itself can call back into WebAssembly to get the
          // current pthread_self() pointer).
          Asyncify.state = Asyncify.State.Normal;
          
          // Keep the runtime alive so that a re-wind can be done later.
          runAndAbortIfError(_asyncify_stop_unwind);
          if (typeof Fibers != 'undefined') {
            Fibers.trampoline();
          }
        }
      },
  whenDone() {
        return new Promise((resolve, reject) => {
          Asyncify.asyncPromiseHandlers = { resolve, reject };
        });
      },
  allocateData() {
        // An asyncify data structure has three fields:
        //  0  current stack pos
        //  4  max stack pos
        //  8  id of function at bottom of the call stack (callStackIdToName[id] == name of js function)
        //
        // The Asyncify ABI only interprets the first two fields, the rest is for the runtime.
        // We also embed a stack in the same memory region here, right next to the structure.
        // This struct is also defined as asyncify_data_t in emscripten/fiber.h
        var ptr = _malloc(12 + Asyncify.StackSize);
        Asyncify.setDataHeader(ptr, ptr + 12, Asyncify.StackSize);
        Asyncify.setDataRewindFunc(ptr);
        return ptr;
      },
  setDataHeader(ptr, stack, stackSize) {
        HEAPU32[((ptr)>>2)] = stack;
        HEAPU32[(((ptr)+(4))>>2)] = stack + stackSize;
      },
  setDataRewindFunc(ptr) {
        var bottomOfCallStack = Asyncify.exportCallStack[0];
        var rewindId = Asyncify.getCallStackId(bottomOfCallStack);
        HEAP32[(((ptr)+(8))>>2)] = rewindId;
      },
  getDataRewindFuncName(ptr) {
        var id = HEAP32[(((ptr)+(8))>>2)];
        var name = Asyncify.callStackIdToName[id];
        return name;
      },
  getDataRewindFunc(name) {
        var func = wasmExports[name];
        return func;
      },
  doRewind(ptr) {
        var name = Asyncify.getDataRewindFuncName(ptr);
        var func = Asyncify.getDataRewindFunc(name);
        // Once we have rewound and the stack we no longer need to artificially
        // keep the runtime alive.
        
        return func();
      },
  handleSleep(startAsync) {
        if (ABORT) return;
        if (Asyncify.state === Asyncify.State.Normal) {
          // Prepare to sleep. Call startAsync, and see what happens:
          // if the code decided to call our callback synchronously,
          // then no async operation was in fact begun, and we don't
          // need to do anything.
          var reachedCallback = false;
          var reachedAfterCallback = false;
          startAsync((handleSleepReturnValue = 0) => {
            if (ABORT) return;
            Asyncify.handleSleepReturnValue = handleSleepReturnValue;
            reachedCallback = true;
            if (!reachedAfterCallback) {
              // We are happening synchronously, so no need for async.
              return;
            }
            Asyncify.state = Asyncify.State.Rewinding;
            runAndAbortIfError(() => _asyncify_start_rewind(Asyncify.currData));
            if (typeof MainLoop != 'undefined' && MainLoop.func) {
              MainLoop.resume();
            }
            var asyncWasmReturnValue, isError = false;
            try {
              asyncWasmReturnValue = Asyncify.doRewind(Asyncify.currData);
            } catch (err) {
              asyncWasmReturnValue = err;
              isError = true;
            }
            // Track whether the return value was handled by any promise handlers.
            var handled = false;
            if (!Asyncify.currData) {
              // All asynchronous execution has finished.
              // `asyncWasmReturnValue` now contains the final
              // return value of the exported async WASM function.
              //
              // Note: `asyncWasmReturnValue` is distinct from
              // `Asyncify.handleSleepReturnValue`.
              // `Asyncify.handleSleepReturnValue` contains the return
              // value of the last C function to have executed
              // `Asyncify.handleSleep()`, where as `asyncWasmReturnValue`
              // contains the return value of the exported WASM function
              // that may have called C functions that
              // call `Asyncify.handleSleep()`.
              var asyncPromiseHandlers = Asyncify.asyncPromiseHandlers;
              if (asyncPromiseHandlers) {
                Asyncify.asyncPromiseHandlers = null;
                (isError ? asyncPromiseHandlers.reject : asyncPromiseHandlers.resolve)(asyncWasmReturnValue);
                handled = true;
              }
            }
            if (isError && !handled) {
              // If there was an error and it was not handled by now, we have no choice but to
              // rethrow that error into the global scope where it can be caught only by
              // `onerror` or `onunhandledpromiserejection`.
              throw asyncWasmReturnValue;
            }
          });
          reachedAfterCallback = true;
          if (!reachedCallback) {
            // A true async operation was begun; start a sleep.
            Asyncify.state = Asyncify.State.Unwinding;
            // TODO: reuse, don't alloc/free every sleep
            Asyncify.currData = Asyncify.allocateData();
            if (typeof MainLoop != 'undefined' && MainLoop.func) {
              MainLoop.pause();
            }
            runAndAbortIfError(() => _asyncify_start_unwind(Asyncify.currData));
          }
        } else if (Asyncify.state === Asyncify.State.Rewinding) {
          // Stop a resume.
          Asyncify.state = Asyncify.State.Normal;
          runAndAbortIfError(_asyncify_stop_rewind);
          _free(Asyncify.currData);
          Asyncify.currData = null;
          // Call all sleep callbacks now that the sleep-resume is all done.
          Asyncify.sleepCallbacks.forEach(callUserCallback);
        } else {
          abort(`invalid state: ${Asyncify.state}`);
        }
        return Asyncify.handleSleepReturnValue;
      },
  handleAsync(startAsync) {
        return Asyncify.handleSleep((wakeUp) => {
          // TODO: add error handling as a second param when handleSleep implements it.
          startAsync().then(wakeUp);
        });
      },
  };

  var getCFunc = (ident) => {
      var func = Module['_' + ident]; // closure exported function
      return func;
    };
  
  var writeArrayToMemory = (array, buffer) => {
      HEAP8.set(array, buffer);
    };
  
  
  
  
  
  
  
  
    /**
     * @param {string|null=} returnType
     * @param {Array=} argTypes
     * @param {Arguments|Array=} args
     * @param {Object=} opts
     */
  var ccall = (ident, returnType, argTypes, args, opts) => {
      // For fast lookup of conversion functions
      var toC = {
        'string': (str) => {
          var ret = 0;
          if (str !== null && str !== undefined && str !== 0) { // null string
            ret = stringToUTF8OnStack(str);
          }
          return ret;
        },
        'array': (arr) => {
          var ret = stackAlloc(arr.length);
          writeArrayToMemory(arr, ret);
          return ret;
        }
      };
  
      function convertReturnValue(ret) {
        if (returnType === 'string') {
          return UTF8ToString(ret);
        }
        if (returnType === 'boolean') return Boolean(ret);
        return ret;
      }
  
      var func = getCFunc(ident);
      var cArgs = [];
      var stack = 0;
      if (args) {
        for (var i = 0; i < args.length; i++) {
          var converter = toC[argTypes[i]];
          if (converter) {
            if (stack === 0) stack = stackSave();
            cArgs[i] = converter(args[i]);
          } else {
            cArgs[i] = args[i];
          }
        }
      }
      // Data for a previous async operation that was in flight before us.
      var previousAsync = Asyncify.currData;
      var ret = func(...cArgs);
      function onDone(ret) {
        runtimeKeepalivePop();
        if (stack !== 0) stackRestore(stack);
        return convertReturnValue(ret);
      }
    var asyncMode = opts?.async;
  
      // Keep the runtime alive through all calls. Note that this call might not be
      // async, but for simplicity we push and pop in all calls.
      runtimeKeepalivePush();
      if (Asyncify.currData != previousAsync) {
        // This is a new async operation. The wasm is paused and has unwound its stack.
        // We need to return a Promise that resolves the return value
        // once the stack is rewound and execution finishes.
        return Asyncify.whenDone().then(onDone);
      }
  
      ret = onDone(ret);
      // If this is an async ccall, ensure we return a promise
      if (asyncMode) return Promise.resolve(ret);
      return ret;
    };

  
  
    /**
     * @param {string=} returnType
     * @param {Array=} argTypes
     * @param {Object=} opts
     */
  var cwrap = (ident, returnType, argTypes, opts) => {
      // When the function takes numbers and returns a number, we can just return
      // the original function
      var numericArgs = !argTypes || argTypes.every((type) => type === 'number' || type === 'boolean');
      var numericRet = returnType !== 'string';
      if (numericRet && numericArgs && !opts) {
        return getCFunc(ident);
      }
      return (...args) => ccall(ident, returnType, argTypes, args, opts);
    };
var wasmImports = {
  /** @export */
  __assert_fail: ___assert_fail,
  /** @export */
  __cxa_throw: ___cxa_throw,
  /** @export */
  _abort_js: __abort_js,
  /** @export */
  _emscripten_runtime_keepalive_clear: __emscripten_runtime_keepalive_clear,
  /** @export */
  _localtime_js: __localtime_js,
  /** @export */
  _setitimer_js: __setitimer_js,
  /** @export */
  _tzset_js: __tzset_js,
  /** @export */
  clock_time_get: _clock_time_get,
  /** @export */
  emscripten_has_asyncify: _emscripten_has_asyncify,
  /** @export */
  emscripten_resize_heap: _emscripten_resize_heap,
  /** @export */
  emscripten_sleep: _emscripten_sleep,
  /** @export */
  emwgpuAdapterRequestDevice: _emwgpuAdapterRequestDevice,
  /** @export */
  emwgpuBufferDestroy: _emwgpuBufferDestroy,
  /** @export */
  emwgpuBufferGetConstMappedRange: _emwgpuBufferGetConstMappedRange,
  /** @export */
  emwgpuBufferMapAsync: _emwgpuBufferMapAsync,
  /** @export */
  emwgpuBufferUnmap: _emwgpuBufferUnmap,
  /** @export */
  emwgpuDelete: _emwgpuDelete,
  /** @export */
  emwgpuDeviceCreateBuffer: _emwgpuDeviceCreateBuffer,
  /** @export */
  emwgpuDeviceCreateShaderModule: _emwgpuDeviceCreateShaderModule,
  /** @export */
  emwgpuDeviceDestroy: _emwgpuDeviceDestroy,
  /** @export */
  emwgpuInstanceRequestAdapter: _emwgpuInstanceRequestAdapter,
  /** @export */
  environ_get: _environ_get,
  /** @export */
  environ_sizes_get: _environ_sizes_get,
  /** @export */
  fd_close: _fd_close,
  /** @export */
  fd_read: _fd_read,
  /** @export */
  fd_seek: _fd_seek,
  /** @export */
  fd_write: _fd_write,
  /** @export */
  proc_exit: _proc_exit,
  /** @export */
  wgpuCommandEncoderBeginComputePass: _wgpuCommandEncoderBeginComputePass,
  /** @export */
  wgpuCommandEncoderCopyBufferToBuffer: _wgpuCommandEncoderCopyBufferToBuffer,
  /** @export */
  wgpuCommandEncoderFinish: _wgpuCommandEncoderFinish,
  /** @export */
  wgpuComputePassEncoderDispatchWorkgroups: _wgpuComputePassEncoderDispatchWorkgroups,
  /** @export */
  wgpuComputePassEncoderEnd: _wgpuComputePassEncoderEnd,
  /** @export */
  wgpuComputePassEncoderSetBindGroup: _wgpuComputePassEncoderSetBindGroup,
  /** @export */
  wgpuComputePassEncoderSetPipeline: _wgpuComputePassEncoderSetPipeline,
  /** @export */
  wgpuDeviceCreateBindGroup: _wgpuDeviceCreateBindGroup,
  /** @export */
  wgpuDeviceCreateBindGroupLayout: _wgpuDeviceCreateBindGroupLayout,
  /** @export */
  wgpuDeviceCreateCommandEncoder: _wgpuDeviceCreateCommandEncoder,
  /** @export */
  wgpuDeviceCreateComputePipeline: _wgpuDeviceCreateComputePipeline,
  /** @export */
  wgpuDeviceCreatePipelineLayout: _wgpuDeviceCreatePipelineLayout,
  /** @export */
  wgpuQueueSubmit: _wgpuQueueSubmit,
  /** @export */
  wgpuQueueWriteBuffer: _wgpuQueueWriteBuffer
};
var wasmExports;
createWasm();
var ___wasm_call_ctors = () => (___wasm_call_ctors = wasmExports['__wasm_call_ctors'])();
var _mgpuInitializeContext = Module['_mgpuInitializeContext'] = () => (_mgpuInitializeContext = Module['_mgpuInitializeContext'] = wasmExports['mgpuInitializeContext'])();
var _mgpuInitializeContextAsync = Module['_mgpuInitializeContextAsync'] = (a0) => (_mgpuInitializeContextAsync = Module['_mgpuInitializeContextAsync'] = wasmExports['mgpuInitializeContextAsync'])(a0);
var _mgpuDestroyContext = Module['_mgpuDestroyContext'] = () => (_mgpuDestroyContext = Module['_mgpuDestroyContext'] = wasmExports['mgpuDestroyContext'])();
var _mgpuCreateComputeShader = Module['_mgpuCreateComputeShader'] = () => (_mgpuCreateComputeShader = Module['_mgpuCreateComputeShader'] = wasmExports['mgpuCreateComputeShader'])();
var _mgpuDestroyComputeShader = Module['_mgpuDestroyComputeShader'] = (a0) => (_mgpuDestroyComputeShader = Module['_mgpuDestroyComputeShader'] = wasmExports['mgpuDestroyComputeShader'])(a0);
var _mgpuLoadKernel = Module['_mgpuLoadKernel'] = (a0, a1) => (_mgpuLoadKernel = Module['_mgpuLoadKernel'] = wasmExports['mgpuLoadKernel'])(a0, a1);
var _mgpuHasKernel = Module['_mgpuHasKernel'] = (a0) => (_mgpuHasKernel = Module['_mgpuHasKernel'] = wasmExports['mgpuHasKernel'])(a0);
var _mgpuCreateBuffer = Module['_mgpuCreateBuffer'] = (a0, a1) => (_mgpuCreateBuffer = Module['_mgpuCreateBuffer'] = wasmExports['mgpuCreateBuffer'])(a0, a1);
var _mgpuDestroyBuffer = Module['_mgpuDestroyBuffer'] = (a0) => (_mgpuDestroyBuffer = Module['_mgpuDestroyBuffer'] = wasmExports['mgpuDestroyBuffer'])(a0);
var _mgpuSetBuffer = Module['_mgpuSetBuffer'] = (a0, a1, a2) => (_mgpuSetBuffer = Module['_mgpuSetBuffer'] = wasmExports['mgpuSetBuffer'])(a0, a1, a2);
var _mgpuDispatch = Module['_mgpuDispatch'] = (a0, a1, a2, a3) => (_mgpuDispatch = Module['_mgpuDispatch'] = wasmExports['mgpuDispatch'])(a0, a1, a2, a3);
var _mgpuDispatchAsync = Module['_mgpuDispatchAsync'] = (a0, a1, a2, a3, a4) => (_mgpuDispatchAsync = Module['_mgpuDispatchAsync'] = wasmExports['mgpuDispatchAsync'])(a0, a1, a2, a3, a4);
var _mgpuReadSync = Module['_mgpuReadSync'] = (a0, a1, a2, a3) => (_mgpuReadSync = Module['_mgpuReadSync'] = wasmExports['mgpuReadSync'])(a0, a1, a2, a3);
var _mgpuReadAsyncFloat = Module['_mgpuReadAsyncFloat'] = (a0, a1, a2, a3, a4) => (_mgpuReadAsyncFloat = Module['_mgpuReadAsyncFloat'] = wasmExports['mgpuReadAsyncFloat'])(a0, a1, a2, a3, a4);
var _mgpuReadAsyncInt8 = Module['_mgpuReadAsyncInt8'] = (a0, a1, a2, a3, a4) => (_mgpuReadAsyncInt8 = Module['_mgpuReadAsyncInt8'] = wasmExports['mgpuReadAsyncInt8'])(a0, a1, a2, a3, a4);
var _mgpuReadAsyncUint8 = Module['_mgpuReadAsyncUint8'] = (a0, a1, a2, a3, a4) => (_mgpuReadAsyncUint8 = Module['_mgpuReadAsyncUint8'] = wasmExports['mgpuReadAsyncUint8'])(a0, a1, a2, a3, a4);
var _mgpuReadAsyncInt16 = Module['_mgpuReadAsyncInt16'] = (a0, a1, a2, a3, a4) => (_mgpuReadAsyncInt16 = Module['_mgpuReadAsyncInt16'] = wasmExports['mgpuReadAsyncInt16'])(a0, a1, a2, a3, a4);
var _mgpuReadAsyncUint16 = Module['_mgpuReadAsyncUint16'] = (a0, a1, a2, a3, a4) => (_mgpuReadAsyncUint16 = Module['_mgpuReadAsyncUint16'] = wasmExports['mgpuReadAsyncUint16'])(a0, a1, a2, a3, a4);
var _mgpuReadAsyncInt32 = Module['_mgpuReadAsyncInt32'] = (a0, a1, a2, a3, a4) => (_mgpuReadAsyncInt32 = Module['_mgpuReadAsyncInt32'] = wasmExports['mgpuReadAsyncInt32'])(a0, a1, a2, a3, a4);
var _mgpuReadAsyncUint32 = Module['_mgpuReadAsyncUint32'] = (a0, a1, a2, a3, a4) => (_mgpuReadAsyncUint32 = Module['_mgpuReadAsyncUint32'] = wasmExports['mgpuReadAsyncUint32'])(a0, a1, a2, a3, a4);
var _mgpuReadAsyncInt64 = Module['_mgpuReadAsyncInt64'] = (a0, a1, a2, a3, a4) => (_mgpuReadAsyncInt64 = Module['_mgpuReadAsyncInt64'] = wasmExports['mgpuReadAsyncInt64'])(a0, a1, a2, a3, a4);
var _mgpuReadAsyncUint64 = Module['_mgpuReadAsyncUint64'] = (a0, a1, a2, a3, a4) => (_mgpuReadAsyncUint64 = Module['_mgpuReadAsyncUint64'] = wasmExports['mgpuReadAsyncUint64'])(a0, a1, a2, a3, a4);
var _mgpuWriteFloat = Module['_mgpuWriteFloat'] = (a0, a1, a2) => (_mgpuWriteFloat = Module['_mgpuWriteFloat'] = wasmExports['mgpuWriteFloat'])(a0, a1, a2);
var _mgpuReadAsyncDouble = Module['_mgpuReadAsyncDouble'] = (a0, a1, a2, a3, a4) => (_mgpuReadAsyncDouble = Module['_mgpuReadAsyncDouble'] = wasmExports['mgpuReadAsyncDouble'])(a0, a1, a2, a3, a4);
var _mgpuReadSyncInt8 = Module['_mgpuReadSyncInt8'] = (a0, a1, a2, a3) => (_mgpuReadSyncInt8 = Module['_mgpuReadSyncInt8'] = wasmExports['mgpuReadSyncInt8'])(a0, a1, a2, a3);
var _mgpuReadSyncUint8 = Module['_mgpuReadSyncUint8'] = (a0, a1, a2, a3) => (_mgpuReadSyncUint8 = Module['_mgpuReadSyncUint8'] = wasmExports['mgpuReadSyncUint8'])(a0, a1, a2, a3);
var _mgpuReadSyncInt16 = Module['_mgpuReadSyncInt16'] = (a0, a1, a2, a3) => (_mgpuReadSyncInt16 = Module['_mgpuReadSyncInt16'] = wasmExports['mgpuReadSyncInt16'])(a0, a1, a2, a3);
var _mgpuReadSyncUint16 = Module['_mgpuReadSyncUint16'] = (a0, a1, a2, a3) => (_mgpuReadSyncUint16 = Module['_mgpuReadSyncUint16'] = wasmExports['mgpuReadSyncUint16'])(a0, a1, a2, a3);
var _mgpuReadSyncInt32 = Module['_mgpuReadSyncInt32'] = (a0, a1, a2, a3) => (_mgpuReadSyncInt32 = Module['_mgpuReadSyncInt32'] = wasmExports['mgpuReadSyncInt32'])(a0, a1, a2, a3);
var _mgpuReadSyncUint32 = Module['_mgpuReadSyncUint32'] = (a0, a1, a2, a3) => (_mgpuReadSyncUint32 = Module['_mgpuReadSyncUint32'] = wasmExports['mgpuReadSyncUint32'])(a0, a1, a2, a3);
var _mgpuReadSyncInt64 = Module['_mgpuReadSyncInt64'] = (a0, a1, a2, a3) => (_mgpuReadSyncInt64 = Module['_mgpuReadSyncInt64'] = wasmExports['mgpuReadSyncInt64'])(a0, a1, a2, a3);
var _mgpuReadSyncUint64 = Module['_mgpuReadSyncUint64'] = (a0, a1, a2, a3) => (_mgpuReadSyncUint64 = Module['_mgpuReadSyncUint64'] = wasmExports['mgpuReadSyncUint64'])(a0, a1, a2, a3);
var _mgpuReadSyncFloat32 = Module['_mgpuReadSyncFloat32'] = (a0, a1, a2, a3) => (_mgpuReadSyncFloat32 = Module['_mgpuReadSyncFloat32'] = wasmExports['mgpuReadSyncFloat32'])(a0, a1, a2, a3);
var _mgpuReadSyncFloat64 = Module['_mgpuReadSyncFloat64'] = (a0, a1, a2, a3) => (_mgpuReadSyncFloat64 = Module['_mgpuReadSyncFloat64'] = wasmExports['mgpuReadSyncFloat64'])(a0, a1, a2, a3);
var _mgpuWriteInt8 = Module['_mgpuWriteInt8'] = (a0, a1, a2) => (_mgpuWriteInt8 = Module['_mgpuWriteInt8'] = wasmExports['mgpuWriteInt8'])(a0, a1, a2);
var _mgpuWriteInt16 = Module['_mgpuWriteInt16'] = (a0, a1, a2) => (_mgpuWriteInt16 = Module['_mgpuWriteInt16'] = wasmExports['mgpuWriteInt16'])(a0, a1, a2);
var _mgpuWriteInt32 = Module['_mgpuWriteInt32'] = (a0, a1, a2) => (_mgpuWriteInt32 = Module['_mgpuWriteInt32'] = wasmExports['mgpuWriteInt32'])(a0, a1, a2);
var _mgpuWriteInt64 = Module['_mgpuWriteInt64'] = (a0, a1, a2) => (_mgpuWriteInt64 = Module['_mgpuWriteInt64'] = wasmExports['mgpuWriteInt64'])(a0, a1, a2);
var _mgpuWriteUint8 = Module['_mgpuWriteUint8'] = (a0, a1, a2) => (_mgpuWriteUint8 = Module['_mgpuWriteUint8'] = wasmExports['mgpuWriteUint8'])(a0, a1, a2);
var _mgpuWriteUint16 = Module['_mgpuWriteUint16'] = (a0, a1, a2) => (_mgpuWriteUint16 = Module['_mgpuWriteUint16'] = wasmExports['mgpuWriteUint16'])(a0, a1, a2);
var _mgpuWriteUint32 = Module['_mgpuWriteUint32'] = (a0, a1, a2) => (_mgpuWriteUint32 = Module['_mgpuWriteUint32'] = wasmExports['mgpuWriteUint32'])(a0, a1, a2);
var _mgpuWriteUint64 = Module['_mgpuWriteUint64'] = (a0, a1, a2) => (_mgpuWriteUint64 = Module['_mgpuWriteUint64'] = wasmExports['mgpuWriteUint64'])(a0, a1, a2);
var _mgpuWriteDouble = Module['_mgpuWriteDouble'] = (a0, a1, a2) => (_mgpuWriteDouble = Module['_mgpuWriteDouble'] = wasmExports['mgpuWriteDouble'])(a0, a1, a2);
var _emwgpuCreateBindGroup = (a0) => (_emwgpuCreateBindGroup = wasmExports['emwgpuCreateBindGroup'])(a0);
var _emwgpuCreateBindGroupLayout = (a0) => (_emwgpuCreateBindGroupLayout = wasmExports['emwgpuCreateBindGroupLayout'])(a0);
var _emwgpuCreateCommandBuffer = (a0) => (_emwgpuCreateCommandBuffer = wasmExports['emwgpuCreateCommandBuffer'])(a0);
var _emwgpuCreateCommandEncoder = (a0) => (_emwgpuCreateCommandEncoder = wasmExports['emwgpuCreateCommandEncoder'])(a0);
var _emwgpuCreateComputePassEncoder = (a0) => (_emwgpuCreateComputePassEncoder = wasmExports['emwgpuCreateComputePassEncoder'])(a0);
var _emwgpuCreateComputePipeline = (a0) => (_emwgpuCreateComputePipeline = wasmExports['emwgpuCreateComputePipeline'])(a0);
var _emwgpuCreatePipelineLayout = (a0) => (_emwgpuCreatePipelineLayout = wasmExports['emwgpuCreatePipelineLayout'])(a0);
var _emwgpuCreateQuerySet = (a0) => (_emwgpuCreateQuerySet = wasmExports['emwgpuCreateQuerySet'])(a0);
var _emwgpuCreateRenderBundle = (a0) => (_emwgpuCreateRenderBundle = wasmExports['emwgpuCreateRenderBundle'])(a0);
var _emwgpuCreateRenderBundleEncoder = (a0) => (_emwgpuCreateRenderBundleEncoder = wasmExports['emwgpuCreateRenderBundleEncoder'])(a0);
var _emwgpuCreateRenderPassEncoder = (a0) => (_emwgpuCreateRenderPassEncoder = wasmExports['emwgpuCreateRenderPassEncoder'])(a0);
var _emwgpuCreateRenderPipeline = (a0) => (_emwgpuCreateRenderPipeline = wasmExports['emwgpuCreateRenderPipeline'])(a0);
var _emwgpuCreateSampler = (a0) => (_emwgpuCreateSampler = wasmExports['emwgpuCreateSampler'])(a0);
var _emwgpuCreateSurface = (a0) => (_emwgpuCreateSurface = wasmExports['emwgpuCreateSurface'])(a0);
var _emwgpuCreateTexture = (a0) => (_emwgpuCreateTexture = wasmExports['emwgpuCreateTexture'])(a0);
var _emwgpuCreateTextureView = (a0) => (_emwgpuCreateTextureView = wasmExports['emwgpuCreateTextureView'])(a0);
var _emwgpuCreateAdapter = (a0) => (_emwgpuCreateAdapter = wasmExports['emwgpuCreateAdapter'])(a0);
var _emwgpuCreateBuffer = (a0, a1) => (_emwgpuCreateBuffer = wasmExports['emwgpuCreateBuffer'])(a0, a1);
var _emwgpuCreateDevice = (a0, a1) => (_emwgpuCreateDevice = wasmExports['emwgpuCreateDevice'])(a0, a1);
var _emwgpuCreateQueue = (a0) => (_emwgpuCreateQueue = wasmExports['emwgpuCreateQueue'])(a0);
var _emwgpuCreateShaderModule = (a0) => (_emwgpuCreateShaderModule = wasmExports['emwgpuCreateShaderModule'])(a0);
var _emwgpuOnCompilationInfoCompleted = (a0, a1, a2) => (_emwgpuOnCompilationInfoCompleted = wasmExports['emwgpuOnCompilationInfoCompleted'])(a0, a1, a2);
var _emwgpuOnCreateComputePipelineCompleted = (a0, a1, a2, a3) => (_emwgpuOnCreateComputePipelineCompleted = wasmExports['emwgpuOnCreateComputePipelineCompleted'])(a0, a1, a2, a3);
var _emwgpuOnCreateRenderPipelineCompleted = (a0, a1, a2, a3) => (_emwgpuOnCreateRenderPipelineCompleted = wasmExports['emwgpuOnCreateRenderPipelineCompleted'])(a0, a1, a2, a3);
var _emwgpuOnDeviceLostCompleted = (a0, a1, a2) => (_emwgpuOnDeviceLostCompleted = wasmExports['emwgpuOnDeviceLostCompleted'])(a0, a1, a2);
var _emwgpuOnMapAsyncCompleted = (a0, a1, a2) => (_emwgpuOnMapAsyncCompleted = wasmExports['emwgpuOnMapAsyncCompleted'])(a0, a1, a2);
var _emwgpuOnPopErrorScopeCompleted = (a0, a1, a2, a3) => (_emwgpuOnPopErrorScopeCompleted = wasmExports['emwgpuOnPopErrorScopeCompleted'])(a0, a1, a2, a3);
var _emwgpuOnRequestAdapterCompleted = (a0, a1, a2, a3) => (_emwgpuOnRequestAdapterCompleted = wasmExports['emwgpuOnRequestAdapterCompleted'])(a0, a1, a2, a3);
var _emwgpuOnRequestDeviceCompleted = (a0, a1, a2, a3) => (_emwgpuOnRequestDeviceCompleted = wasmExports['emwgpuOnRequestDeviceCompleted'])(a0, a1, a2, a3);
var _emwgpuOnWorkDoneCompleted = (a0, a1) => (_emwgpuOnWorkDoneCompleted = wasmExports['emwgpuOnWorkDoneCompleted'])(a0, a1);
var _emwgpuOnUncapturedError = (a0, a1, a2) => (_emwgpuOnUncapturedError = wasmExports['emwgpuOnUncapturedError'])(a0, a1, a2);
var _free = Module['_free'] = (a0) => (_free = Module['_free'] = wasmExports['free'])(a0);
var _memcpy = Module['_memcpy'] = (a0, a1, a2) => (_memcpy = Module['_memcpy'] = wasmExports['memcpy'])(a0, a1, a2);
var __emscripten_timeout = (a0, a1) => (__emscripten_timeout = wasmExports['_emscripten_timeout'])(a0, a1);
var _memalign = (a0, a1) => (_memalign = wasmExports['memalign'])(a0, a1);
var _malloc = Module['_malloc'] = (a0) => (_malloc = Module['_malloc'] = wasmExports['malloc'])(a0);
var __emscripten_stack_restore = (a0) => (__emscripten_stack_restore = wasmExports['_emscripten_stack_restore'])(a0);
var __emscripten_stack_alloc = (a0) => (__emscripten_stack_alloc = wasmExports['_emscripten_stack_alloc'])(a0);
var _emscripten_stack_get_current = () => (_emscripten_stack_get_current = wasmExports['emscripten_stack_get_current'])();
var ___cxa_increment_exception_refcount = (a0) => (___cxa_increment_exception_refcount = wasmExports['__cxa_increment_exception_refcount'])(a0);
var dynCall_ii = Module['dynCall_ii'] = (a0, a1) => (dynCall_ii = Module['dynCall_ii'] = wasmExports['dynCall_ii'])(a0, a1);
var dynCall_vi = Module['dynCall_vi'] = (a0, a1) => (dynCall_vi = Module['dynCall_vi'] = wasmExports['dynCall_vi'])(a0, a1);
var dynCall_viiii = Module['dynCall_viiii'] = (a0, a1, a2, a3, a4) => (dynCall_viiii = Module['dynCall_viiii'] = wasmExports['dynCall_viiii'])(a0, a1, a2, a3, a4);
var dynCall_viiiii = Module['dynCall_viiiii'] = (a0, a1, a2, a3, a4, a5) => (dynCall_viiiii = Module['dynCall_viiiii'] = wasmExports['dynCall_viiiii'])(a0, a1, a2, a3, a4, a5);
var dynCall_vii = Module['dynCall_vii'] = (a0, a1, a2) => (dynCall_vii = Module['dynCall_vii'] = wasmExports['dynCall_vii'])(a0, a1, a2);
var dynCall_iii = Module['dynCall_iii'] = (a0, a1, a2) => (dynCall_iii = Module['dynCall_iii'] = wasmExports['dynCall_iii'])(a0, a1, a2);
var dynCall_v = Module['dynCall_v'] = (a0) => (dynCall_v = Module['dynCall_v'] = wasmExports['dynCall_v'])(a0);
var dynCall_iiii = Module['dynCall_iiii'] = (a0, a1, a2, a3) => (dynCall_iiii = Module['dynCall_iiii'] = wasmExports['dynCall_iiii'])(a0, a1, a2, a3);
var dynCall_iiiii = Module['dynCall_iiiii'] = (a0, a1, a2, a3, a4) => (dynCall_iiiii = Module['dynCall_iiiii'] = wasmExports['dynCall_iiiii'])(a0, a1, a2, a3, a4);
var dynCall_viji = Module['dynCall_viji'] = (a0, a1, a2, a3) => (dynCall_viji = Module['dynCall_viji'] = wasmExports['dynCall_viji'])(a0, a1, a2, a3);
var dynCall_iidiiii = Module['dynCall_iidiiii'] = (a0, a1, a2, a3, a4, a5, a6) => (dynCall_iidiiii = Module['dynCall_iidiiii'] = wasmExports['dynCall_iidiiii'])(a0, a1, a2, a3, a4, a5, a6);
var dynCall_jiji = Module['dynCall_jiji'] = (a0, a1, a2, a3) => (dynCall_jiji = Module['dynCall_jiji'] = wasmExports['dynCall_jiji'])(a0, a1, a2, a3);
var dynCall_viijii = Module['dynCall_viijii'] = (a0, a1, a2, a3, a4, a5) => (dynCall_viijii = Module['dynCall_viijii'] = wasmExports['dynCall_viijii'])(a0, a1, a2, a3, a4, a5);
var dynCall_iiiiii = Module['dynCall_iiiiii'] = (a0, a1, a2, a3, a4, a5) => (dynCall_iiiiii = Module['dynCall_iiiiii'] = wasmExports['dynCall_iiiiii'])(a0, a1, a2, a3, a4, a5);
var dynCall_iiiiiiiii = Module['dynCall_iiiiiiiii'] = (a0, a1, a2, a3, a4, a5, a6, a7, a8) => (dynCall_iiiiiiiii = Module['dynCall_iiiiiiiii'] = wasmExports['dynCall_iiiiiiiii'])(a0, a1, a2, a3, a4, a5, a6, a7, a8);
var dynCall_iiiiiii = Module['dynCall_iiiiiii'] = (a0, a1, a2, a3, a4, a5, a6) => (dynCall_iiiiiii = Module['dynCall_iiiiiii'] = wasmExports['dynCall_iiiiiii'])(a0, a1, a2, a3, a4, a5, a6);
var dynCall_iiiiij = Module['dynCall_iiiiij'] = (a0, a1, a2, a3, a4, a5) => (dynCall_iiiiij = Module['dynCall_iiiiij'] = wasmExports['dynCall_iiiiij'])(a0, a1, a2, a3, a4, a5);
var dynCall_iiiiid = Module['dynCall_iiiiid'] = (a0, a1, a2, a3, a4, a5) => (dynCall_iiiiid = Module['dynCall_iiiiid'] = wasmExports['dynCall_iiiiid'])(a0, a1, a2, a3, a4, a5);
var dynCall_iiiiijj = Module['dynCall_iiiiijj'] = (a0, a1, a2, a3, a4, a5, a6) => (dynCall_iiiiijj = Module['dynCall_iiiiijj'] = wasmExports['dynCall_iiiiijj'])(a0, a1, a2, a3, a4, a5, a6);
var dynCall_iiiiiiii = Module['dynCall_iiiiiiii'] = (a0, a1, a2, a3, a4, a5, a6, a7) => (dynCall_iiiiiiii = Module['dynCall_iiiiiiii'] = wasmExports['dynCall_iiiiiiii'])(a0, a1, a2, a3, a4, a5, a6, a7);
var dynCall_iiiiiijj = Module['dynCall_iiiiiijj'] = (a0, a1, a2, a3, a4, a5, a6, a7) => (dynCall_iiiiiijj = Module['dynCall_iiiiiijj'] = wasmExports['dynCall_iiiiiijj'])(a0, a1, a2, a3, a4, a5, a6, a7);
var dynCall_viiiiii = Module['dynCall_viiiiii'] = (a0, a1, a2, a3, a4, a5, a6) => (dynCall_viiiiii = Module['dynCall_viiiiii'] = wasmExports['dynCall_viiiiii'])(a0, a1, a2, a3, a4, a5, a6);
var _asyncify_start_unwind = (a0) => (_asyncify_start_unwind = wasmExports['asyncify_start_unwind'])(a0);
var _asyncify_stop_unwind = () => (_asyncify_stop_unwind = wasmExports['asyncify_stop_unwind'])();
var _asyncify_start_rewind = (a0) => (_asyncify_start_rewind = wasmExports['asyncify_start_rewind'])(a0);
var _asyncify_stop_rewind = () => (_asyncify_stop_rewind = wasmExports['asyncify_stop_rewind'])();


// include: postamble.js
// === Auto-generated postamble setup entry stuff ===

Module['ccall'] = ccall;
Module['cwrap'] = cwrap;


function run() {

  if (runDependencies > 0) {
    dependenciesFulfilled = run;
    return;
  }

  preRun();

  // a preRun added a dependency, run will be called later
  if (runDependencies > 0) {
    dependenciesFulfilled = run;
    return;
  }

  function doRun() {
    // run may have just been called through dependencies being fulfilled just in this very frame,
    // or while the async setStatus time below was happening
    Module['calledRun'] = true;

    if (ABORT) return;

    initRuntime();

    Module['onRuntimeInitialized']?.();

    postRun();
  }

  if (Module['setStatus']) {
    Module['setStatus']('Running...');
    setTimeout(() => {
      setTimeout(() => Module['setStatus'](''), 1);
      doRun();
    }, 1);
  } else
  {
    doRun();
  }
}

if (Module['preInit']) {
  if (typeof Module['preInit'] == 'function') Module['preInit'] = [Module['preInit']];
  while (Module['preInit'].length > 0) {
    Module['preInit'].pop()();
  }
}

run();

// end include: postamble.js

