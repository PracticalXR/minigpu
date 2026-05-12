// iOS stub — IOSurface/CVPixelBuffer zero-copy path is not yet implemented.
import Flutter
import UIKit

public class MinigpuViewPlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(
      name: "minigpu_view",
      binaryMessenger: registrar.messenger())
    let instance = MinigpuViewPlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    result(FlutterError(
      code: "unsupported",
      message: "iOS GPU preview not yet implemented",
      details: nil))
  }
}
