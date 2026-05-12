// macOS stub — IOSurface zero-copy path is not yet implemented.
import Cocoa
import FlutterMacOS

public class MinigpuViewPlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(
      name: "minigpu_view",
      binaryMessenger: registrar.messenger)
    let instance = MinigpuViewPlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    result(FlutterError(
      code: "unsupported",
      message: "macOS GPU preview not yet implemented",
      details: nil))
  }
}
